// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"

#include "timer.h"
#include "pq_flash_index.h"
#include "cosine_similarity.h"

#ifdef _WINDOWS
#include "windows_aligned_file_reader.h"
#else
#include "linux_aligned_file_reader.h"
#endif

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(_u64))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(_u32))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

// sector # on disk where node_id is present with in the graph part
#define NODE_SECTOR_NO(node_id) (((_u64)(node_id)) / nnodes_per_sector + 1)

// obtains region of sector containing node
#define OFFSET_TO_NODE(sector_buf, node_id) ((char *)sector_buf + (((_u64)node_id) % nnodes_per_sector) * max_node_len)

// returns region of `node_buf` containing [NNBRS][NBR_ID(_u32)]
#define OFFSET_TO_NODE_NHOOD(node_buf) (unsigned *)((char *)node_buf + disk_bytes_per_point)

// returns region of `node_buf` containing [COORD(T)]
#define OFFSET_TO_NODE_COORDS(node_buf) (T *)(node_buf)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) (((_u64)(id)) / nvecs_per_sector + reorder_data_start_sector)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) ((((_u64)(id)) % nvecs_per_sector) * data_dim * sizeof(float))

namespace diskann
{

template <typename T, typename LabelT>
PQFlashIndex<T, LabelT>::PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader, diskann::Metric m)
    : reader(fileReader), metric(m)
{
    if (m == diskann::Metric::COSINE || m == diskann::Metric::INNER_PRODUCT)
    {
        if (std::is_floating_point<T>::value)
        {
            diskann::cout << "Cosine metric chosen for (normalized) float data."
                             "Changing distance to L2 to boost accuracy."
                          << std::endl;
            metric = diskann::Metric::L2;
        }
        else
        {
            diskann::cerr << "WARNING: Cannot normalize integral data types."
                          << " This may result in erroneous results or poor recall."
                          << " Consider using L2 distance with integral data types." << std::endl;
        }
    }

    this->dist_cmp.reset(diskann::get_distance_function<T>(metric));
    this->dist_cmp_float.reset(diskann::get_distance_function<float>(metric));
}

template <typename T, typename LabelT> PQFlashIndex<T, LabelT>::~PQFlashIndex()
{
#ifndef EXEC_ENV_OLS
    if (data != nullptr)
    {
        delete[] data;
    }
#endif

    if (centroid_data != nullptr)
        aligned_free(centroid_data);
    // delete backing bufs for nhood and coord cache
    if (nhood_cache_buf != nullptr)
    {
        delete[] nhood_cache_buf;
        diskann::aligned_free(coord_cache_buf);
    }

    if (load_flag)
    {
        diskann::cout << "Clearing scratch" << std::endl;
        ScratchStoreManager<SSDThreadData<T>> manager(this->thread_data);
        manager.destroy();
        this->reader->deregister_all_threads();
        reader->close();
    }
    if (_pts_to_label_offsets != nullptr)
    {
        delete[] _pts_to_label_offsets;
    }

    if (_pts_to_labels != nullptr)
    {
        delete[] _pts_to_labels;
    }
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::setup_thread_data(_u64 nthreads, _u64 visited_reserve)
{
    diskann::cout << "Setting up thread-specific contexts for nthreads: " << nthreads << std::endl;
// omp parallel for to generate unique thread IDs
#pragma omp parallel for num_threads((int)nthreads)
    for (_s64 thread = 0; thread < (_s64)nthreads; thread++)
    {
#pragma omp critical
        {
            SSDThreadData<T> *data = new SSDThreadData<T>(this->aligned_dim, visited_reserve);
            this->reader->register_thread();
            data->ctx = this->reader->get_ctx();
            this->thread_data.push(data);
        }
    }
    load_flag = true;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::load_cache_list(std::vector<uint32_t> &node_list)
{
    diskann::cout << "Loading the cache list into memory.." << std::flush;
    _u64 num_cached_nodes = node_list.size();

    // borrow thread data
    ScratchStoreManager<SSDThreadData<T>> manager(this->thread_data);
    auto this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;

    nhood_cache_buf = new unsigned[num_cached_nodes * (max_degree + 1)];
    memset(nhood_cache_buf, 0, num_cached_nodes * (max_degree + 1));

    _u64 coord_cache_buf_len = num_cached_nodes * aligned_dim;
    diskann::alloc_aligned((void **)&coord_cache_buf, coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    size_t BLOCK_SIZE = 8;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);

    for (_u64 block = 0; block < num_blocks; block++)
    {
        _u64 start_idx = block * BLOCK_SIZE;
        _u64 end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);
        std::vector<AlignedRead> read_reqs;
        std::vector<std::pair<_u32, char *>> nhoods;
        for (_u64 node_idx = start_idx; node_idx < end_idx; node_idx++)
        {
            AlignedRead read;
            char *buf = nullptr;
            alloc_aligned((void **)&buf, SECTOR_LEN, SECTOR_LEN);
            nhoods.push_back(std::make_pair(node_list[node_idx], buf));
            read.len = SECTOR_LEN;
            read.buf = buf;
            read.offset = NODE_SECTOR_NO(node_list[node_idx]) * SECTOR_LEN;
            read_reqs.push_back(read);
        }

        reader->read(read_reqs, ctx);

        _u64 node_idx = start_idx;
        for (_u32 i = 0; i < read_reqs.size(); i++)
        {
#if defined(_WINDOWS) && defined(USE_BING_INFRA) // this block is to handle failed reads in
                                                 // production settings
            if ((*ctx.m_pRequestsStatus)[i] != IOContext::READ_SUCCESS)
            {
                continue;
            }
#endif
            auto &nhood = nhoods[i];
            char *node_buf = OFFSET_TO_NODE(nhood.second, nhood.first);
            T *node_coords = OFFSET_TO_NODE_COORDS(node_buf);
            T *cached_coords = coord_cache_buf + node_idx * aligned_dim;
            memcpy(cached_coords, node_coords, disk_bytes_per_point);
            coord_cache.insert(std::make_pair(nhood.first, cached_coords));

            // insert node nhood into nhood_cache
            unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);

            auto nnbrs = *node_nhood;
            unsigned *nbrs = node_nhood + 1;
            std::pair<_u32, unsigned *> cnhood;
            cnhood.first = nnbrs;
            cnhood.second = nhood_cache_buf + node_idx * (max_degree + 1);
            memcpy(cnhood.second, nbrs, nnbrs * sizeof(unsigned));
            nhood_cache.insert(std::make_pair(nhood.first, cnhood));
            aligned_free(nhood.second);
            node_idx++;
        }
    }
    diskann::cout << "..done." << std::endl;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                      _u64 l_search, _u64 beamwidth,
                                                                      _u64 num_nodes_to_cache, uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#else
template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::generate_cache_list_from_sample_queries(std::string sample_bin, _u64 l_search,
                                                                      _u64 beamwidth, _u64 num_nodes_to_cache,
                                                                      uint32_t nthreads,
                                                                      std::vector<uint32_t> &node_list)
{
#endif
    this->count_visited_nodes = true;
    this->node_visit_counter.clear();
    this->node_visit_counter.resize(this->num_points);
    for (_u32 i = 0; i < node_visit_counter.size(); i++)
    {
        this->node_visit_counter[i].first = i;
        this->node_visit_counter[i].second = 0;
    }

    _u64 sample_num, sample_dim, sample_aligned_dim;
    T *samples;

#ifdef EXEC_ENV_OLS
    if (files.fileExists(sample_bin))
    {
        diskann::load_aligned_bin<T>(files, sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#else
    if (file_exists(sample_bin))
    {
        diskann::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
    }
#endif
    else
    {
        diskann::cerr << "Sample bin file not found. Not generating cache." << std::endl;
        return;
    }

    std::vector<uint64_t> tmp_result_ids_64(sample_num, 0);
    std::vector<float> tmp_result_dists(sample_num, 0);

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (_s64 i = 0; i < (int64_t)sample_num; i++)
    {
        cached_beam_search(samples + (i * sample_aligned_dim), 1, l_search, tmp_result_ids_64.data() + (i * 1),
                           tmp_result_dists.data() + (i * 1), beamwidth);
    }

    std::sort(this->node_visit_counter.begin(), node_visit_counter.end(),
              [](std::pair<_u32, _u32> &left, std::pair<_u32, _u32> &right) { return left.second > right.second; });
    node_list.clear();
    node_list.shrink_to_fit();
    node_list.reserve(num_nodes_to_cache);
    for (_u64 i = 0; i < num_nodes_to_cache; i++)
    {
        node_list.push_back(this->node_visit_counter[i].first);
    }
    this->count_visited_nodes = false;

    diskann::aligned_free(samples);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cache_bfs_levels(_u64 num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                               const bool shuffle)
{
    std::random_device rng;
    std::mt19937 urng(rng());

    tsl::robin_set<uint32_t> node_set;

    // Do not cache more than 10% of the nodes in the index
    _u64 tenp_nodes = (_u64)(std::round(this->num_points * 0.1));
    if (num_nodes_to_cache > tenp_nodes)
    {
        diskann::cout << "Reducing nodes to cache from: " << num_nodes_to_cache << " to: " << tenp_nodes
                      << "(10 percent of total nodes:" << this->num_points << ")" << std::endl;
        num_nodes_to_cache = tenp_nodes == 0 ? 1 : tenp_nodes;
    }
    diskann::cout << "Caching " << num_nodes_to_cache << "..." << std::endl;

    // borrow thread data
    ScratchStoreManager<SSDThreadData<T>> manager(this->thread_data);
    auto this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;

    std::unique_ptr<tsl::robin_set<unsigned>> cur_level, prev_level;
    cur_level = std::make_unique<tsl::robin_set<unsigned>>();
    prev_level = std::make_unique<tsl::robin_set<unsigned>>();

    for (_u64 miter = 0; miter < num_medoids; miter++)
    {
        cur_level->insert(medoids[miter]);
    }

    _u64 lvl = 1;
    uint64_t prev_node_set_size = 0;
    while ((node_set.size() + cur_level->size() < num_nodes_to_cache) && cur_level->size() != 0)
    {
        // swap prev_level and cur_level
        std::swap(prev_level, cur_level);
        // clear cur_level
        cur_level->clear();

        std::vector<unsigned> nodes_to_expand;

        for (const unsigned &id : *prev_level)
        {
            if (node_set.find(id) != node_set.end())
            {
                continue;
            }
            node_set.insert(id);
            nodes_to_expand.push_back(id);
        }

        if (shuffle)
            std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);
        else
            std::sort(nodes_to_expand.begin(), nodes_to_expand.end());

        diskann::cout << "Level: " << lvl << std::flush;
        bool finish_flag = false;

        uint64_t BLOCK_SIZE = 1024;
        uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), BLOCK_SIZE);
        for (size_t block = 0; block < nblocks && !finish_flag; block++)
        {
            diskann::cout << "." << std::flush;
            size_t start = block * BLOCK_SIZE;
            size_t end = (std::min)((block + 1) * BLOCK_SIZE, nodes_to_expand.size());
            std::vector<AlignedRead> read_reqs;
            std::vector<std::pair<_u32, char *>> nhoods;
            for (size_t cur_pt = start; cur_pt < end; cur_pt++)
            {
                char *buf = nullptr;
                alloc_aligned((void **)&buf, SECTOR_LEN, SECTOR_LEN);
                nhoods.emplace_back(nodes_to_expand[cur_pt], buf);
                AlignedRead read;
                read.len = SECTOR_LEN;
                read.buf = buf;
                read.offset = NODE_SECTOR_NO(nodes_to_expand[cur_pt]) * SECTOR_LEN;
                read_reqs.push_back(read);
            }

            // issue read requests
            reader->read(read_reqs, ctx);

            // process each nhood buf
            for (_u32 i = 0; i < read_reqs.size(); i++)
            {
#if defined(_WINDOWS) && defined(USE_BING_INFRA) // this block is to handle read failures in
                                                 // production settings
                if ((*ctx.m_pRequestsStatus)[i] != IOContext::READ_SUCCESS)
                {
                    continue;
                }
#endif
                auto &nhood = nhoods[i];

                // insert node coord into coord_cache
                char *node_buf = OFFSET_TO_NODE(nhood.second, nhood.first);
                unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);
                _u64 nnbrs = (_u64)*node_nhood;
                unsigned *nbrs = node_nhood + 1;
                // explore next level
                for (_u64 j = 0; j < nnbrs && !finish_flag; j++)
                {
                    if (node_set.find(nbrs[j]) == node_set.end())
                    {
                        cur_level->insert(nbrs[j]);
                    }
                    if (cur_level->size() + node_set.size() >= num_nodes_to_cache)
                    {
                        finish_flag = true;
                    }
                }
                aligned_free(nhood.second);
            }
        }

        diskann::cout << ". #nodes: " << node_set.size() - prev_node_set_size
                      << ", #nodes thus far: " << node_list.size() << std::endl;
        prev_node_set_size = node_set.size();
        lvl++;
    }

    assert(node_set.size() + cur_level->size() == num_nodes_to_cache || cur_level->size() == 0);

    node_list.clear();
    node_list.reserve(node_set.size() + cur_level->size());
    for (auto node : node_set)
        node_list.push_back(node);
    for (auto node : *cur_level)
        node_list.push_back(node);

    diskann::cout << "Level: " << lvl << std::flush;
    diskann::cout << ". #nodes: " << node_list.size() - prev_node_set_size << ", #nodes thus far: " << node_list.size()
                  << std::endl;
    diskann::cout << "done" << std::endl;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::use_medoids_data_as_centroids()
{
    if (centroid_data != nullptr)
        aligned_free(centroid_data);
    alloc_aligned(((void **)&centroid_data), num_medoids * aligned_dim * sizeof(float), 32);
    std::memset(centroid_data, 0, num_medoids * aligned_dim * sizeof(float));

    // borrow ctx
    ScratchStoreManager<SSDThreadData<T>> manager(this->thread_data);
    auto data = manager.scratch_space();
    IOContext &ctx = data->ctx;
    diskann::cout << "Loading centroid data from medoids vector data of " << num_medoids << " medoid(s)" << std::endl;
    for (uint64_t cur_m = 0; cur_m < num_medoids; cur_m++)
    {
        auto medoid = medoids[cur_m];
        // read medoid nhood
        char *medoid_buf = nullptr;
        alloc_aligned((void **)&medoid_buf, SECTOR_LEN, SECTOR_LEN);
        std::vector<AlignedRead> medoid_read(1);
        medoid_read[0].len = SECTOR_LEN;
        medoid_read[0].buf = medoid_buf;
        medoid_read[0].offset = NODE_SECTOR_NO(medoid) * SECTOR_LEN;
        reader->read(medoid_read, ctx);

        // all data about medoid
        char *medoid_node_buf = OFFSET_TO_NODE(medoid_buf, medoid);

        // add medoid coords to `coord_cache`
        T *medoid_coords = new T[data_dim];
        T *medoid_disk_coords = OFFSET_TO_NODE_COORDS(medoid_node_buf);
        memcpy(medoid_coords, medoid_disk_coords, disk_bytes_per_point);

        if (!use_disk_index_pq)
        {
            for (uint32_t i = 0; i < data_dim; i++)
                centroid_data[cur_m * aligned_dim + i] = medoid_coords[i];
        }
        else
        {
            disk_pq_table.inflate_vector((_u8 *)medoid_coords, (centroid_data + cur_m * aligned_dim));
        }

        aligned_free(medoid_buf);
        delete[] medoid_coords;
    }
}

template <typename T, typename LabelT>
inline int32_t PQFlashIndex<T, LabelT>::get_filter_number(const LabelT &filter_label)
{
    int idx = -1;
    for (_u32 i = 0; i < _filter_list.size(); i++)
    {
        if (_filter_list[i] == filter_label)
        {
            idx = i;
            break;
        }
    }
    return idx;
}

template <typename T, typename LabelT>
std::unordered_map<std::string, LabelT> PQFlashIndex<T, LabelT>::load_label_map(const std::string &labels_map_file)
{
    std::unordered_map<std::string, LabelT> string_to_int_mp;
    std::ifstream map_reader(labels_map_file);
    std::string line, token;
    LabelT token_as_num;
    std::string label_str;
    while (std::getline(map_reader, line))
    {
        std::istringstream iss(line);
        getline(iss, token, '\t');
        label_str = token;
        getline(iss, token, '\t');
        token_as_num = std::stoul(token);
        string_to_int_mp[label_str] = token_as_num;
    }
    return string_to_int_mp;
}

template <typename T, typename LabelT>
LabelT PQFlashIndex<T, LabelT>::get_converted_label(const std::string &filter_label)
{
    if (_label_map.find(filter_label) != _label_map.end())
    {
        return _label_map[filter_label];
    }
    std::stringstream stream;
    stream << "Unable to find label in the Label Map";
    diskann::cerr << stream.str() << std::endl;
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    exit(-1);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::get_label_file_metadata(std::string map_file, _u32 &num_pts, _u32 &num_total_labels)
{
    std::ifstream infile(map_file);
    std::string line, token;
    num_pts = 0;
    num_total_labels = 0;

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        while (getline(iss, token, ','))
        {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
            num_total_labels++;
        }
        num_pts++;
    }

    diskann::cout << "Labels file metadata: num_points: " << num_pts << ", #total_labels: " << num_total_labels
                  << std::endl;
    infile.close();
}

template <typename T, typename LabelT>
inline bool PQFlashIndex<T, LabelT>::point_has_label(_u32 point_id, _u32 label_id)
{
    _u32 start_vec = _pts_to_label_offsets[point_id];
    _u32 num_lbls = _pts_to_labels[start_vec];
    bool ret_val = false;
    for (_u32 i = 0; i < num_lbls; i++)
    {
        if (_pts_to_labels[start_vec + 1 + i] == label_id)
        {
            ret_val = true;
            break;
        }
    }
    return ret_val;
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::parse_label_file(const std::string &label_file, size_t &num_points_labels)
{
    std::ifstream infile(label_file);
    if (infile.fail())
    {
        throw diskann::ANNException(std::string("Failed to open file ") + label_file, -1);
    }

    std::string line, token;
    _u32 line_cnt = 0;

    _u32 num_pts_in_label_file;
    _u32 num_total_labels;
    get_label_file_metadata(label_file, num_pts_in_label_file, num_total_labels);

    _pts_to_label_offsets = new _u32[num_pts_in_label_file];
    _pts_to_labels = new _u32[num_pts_in_label_file + num_total_labels];
    _u32 counter = 0;

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::vector<_u32> lbls(0);

        _pts_to_label_offsets[line_cnt] = counter;
        _u32 &num_lbls_in_cur_pt = _pts_to_labels[counter];
        num_lbls_in_cur_pt = 0;
        counter++;
        getline(iss, token, '\t');
        std::istringstream new_iss(token);
        while (getline(new_iss, token, ','))
        {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
            LabelT token_as_num = std::stoul(token);
            if (_labels.find(token_as_num) == _labels.end())
            {
                _filter_list.emplace_back(token_as_num);
            }
            int32_t filter_num = get_filter_number(token_as_num);
            if (filter_num == -1)
            {
                diskann::cout << "Error!! " << std::endl;
                exit(-1);
            }
            _pts_to_labels[counter++] = filter_num;
            num_lbls_in_cur_pt++;
            _labels.insert(token_as_num);
        }

        if (num_lbls_in_cur_pt == 0)
        {
            diskann::cout << "No label found for point " << line_cnt << std::endl;
            exit(-1);
        }
        line_cnt++;
    }
    infile.close();
    num_points_labels = line_cnt;
}

template <typename T, typename LabelT> void PQFlashIndex<T, LabelT>::set_universal_label(const LabelT &label)
{
    int32_t temp_filter_num = get_filter_number(label);
    if (temp_filter_num == -1) {
      diskann::cout << "Error, could not find universal label."
                    << std::endl;
    } else {
      _use_universal_label = true;
      _universal_filter_num = (_u32) temp_filter_num;
    }
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load(MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix)
{
#else
template <typename T, typename LabelT> int PQFlashIndex<T, LabelT>::load(uint32_t num_threads, const char *index_prefix)
{
#endif
    std::string pq_table_bin = std::string(index_prefix) + "_pq_pivots.bin";
    std::string pq_compressed_vectors = std::string(index_prefix) + "_pq_compressed.bin";
    std::string disk_index_file = std::string(index_prefix) + "_disk.index";
#ifdef EXEC_ENV_OLS
    return load_from_separate_paths(files, num_threads, disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str());
#else
    return load_from_separate_paths(num_threads, disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str());
#endif
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                      const char *index_filepath, const char *pivots_filepath,
                                                      const char *compressed_filepath)
{
#else
template <typename T, typename LabelT>
int PQFlashIndex<T, LabelT>::load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                      const char *pivots_filepath, const char *compressed_filepath)
{
#endif
    std::string pq_table_bin = pivots_filepath;
    std::string pq_compressed_vectors = compressed_filepath;
    std::string disk_index_file = index_filepath;
    std::string medoids_file = std::string(disk_index_file) + "_medoids.bin";
    std::string centroids_file = std::string(disk_index_file) + "_centroids.bin";

    std::string labels_file = std ::string(disk_index_file) + "_labels.txt";
    std::string labels_to_medoids = std ::string(disk_index_file) + "_labels_to_medoids.txt";
    std::string dummy_map_file = std ::string(disk_index_file) + "_dummy_map.txt";
    std::string labels_map_file = std ::string(disk_index_file) + "_labels_map.txt";
    size_t num_pts_in_label_file = 0;

    size_t pq_file_dim, pq_file_num_centroids;
#ifdef EXEC_ENV_OLS
    get_bin_metadata(files, pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#else
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
#endif

    this->disk_index_file = disk_index_file;

    if (pq_file_num_centroids != 256)
    {
        diskann::cout << "Error. Number of PQ centroids is not 256. Exiting." << std::endl;
        return -1;
    }

    this->data_dim = pq_file_dim;
    // will reset later if we use PQ on disk
    this->disk_data_dim = this->data_dim;
    // will change later if we use PQ on disk or if we are using
    // inner product without PQ
    this->disk_bytes_per_point = this->data_dim * sizeof(T);
    this->aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
#ifdef EXEC_ENV_OLS
    diskann::load_bin<_u8>(files, pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#else
    diskann::load_bin<_u8>(pq_compressed_vectors, this->data, npts_u64, nchunks_u64);
#endif

    this->num_points = npts_u64;
    this->n_chunks = nchunks_u64;
    if (file_exists(labels_file))
    {
        parse_label_file(labels_file, num_pts_in_label_file);
        assert(num_pts_in_label_file == this->num_points);
        _label_map = load_label_map(labels_map_file);
        if (file_exists(labels_to_medoids))
        {
            std::ifstream medoid_stream(labels_to_medoids);
            assert(medoid_stream.is_open());
            std::string line, token;

            _filter_to_medoid_id.clear();
            try
            {
                while (std::getline(medoid_stream, line))
                {
                    std::istringstream iss(line);
                    _u32 cnt = 0;
                    _u32 medoid;
                    LabelT label;
                    while (std::getline(iss, token, ','))
                    {
                        if (cnt == 0)
                            label = std::stoul(token);
                        else
                            medoid = (_u32)stoul(token);
                        cnt++;
                    }
                    _filter_to_medoid_id[label] = medoid;
                }
            }
            catch (std::system_error &e)
            {
                throw FileException(labels_to_medoids, e, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
        std::string univ_label_file = std ::string(disk_index_file) + "_universal_label.txt";
        if (file_exists(univ_label_file))
        {
            std::ifstream universal_label_reader(univ_label_file);
            assert(universal_label_reader.is_open());
            std::string univ_label;
            universal_label_reader >> univ_label;
            universal_label_reader.close();
            LabelT label_as_num = std::stoul(univ_label);
            set_universal_label(label_as_num);
        }
        if (file_exists(dummy_map_file))
        {
            std::ifstream dummy_map_stream(dummy_map_file);
            assert(dummy_map_stream.is_open());
            std::string line, token;

            while (std::getline(dummy_map_stream, line))
            {
                std::istringstream iss(line);
                _u32 cnt = 0;
                _u32 dummy_id;
                _u32 real_id;
                while (std::getline(iss, token, ','))
                {
                    if (cnt == 0)
                        dummy_id = (_u32)stoul(token);
                    else
                        real_id = (_u32)stoul(token);
                    cnt++;
                }
                _dummy_pts.insert(dummy_id);
                _has_dummy_pts.insert(real_id);
                _dummy_to_real_map[dummy_id] = real_id;

                if (_real_to_dummy_map.find(real_id) == _real_to_dummy_map.end())
                    _real_to_dummy_map[real_id] = std::vector<_u32>();

                _real_to_dummy_map[real_id].emplace_back(dummy_id);
            }
            dummy_map_stream.close();
            diskann::cout << "Loaded dummy map" << std::endl;
        }
    }

#ifdef EXEC_ENV_OLS
    pq_table.load_pq_centroid_bin(files, pq_table_bin.c_str(), nchunks_u64);
#else
    pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
#endif

    diskann::cout << "Loaded PQ centroids and in-memory compressed vectors. #points: " << num_points
                  << " #dim: " << data_dim << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks << std::endl;

    if (n_chunks > MAX_PQ_CHUNKS)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max PQ bytes for in-memory "
                  "PQ data does not exceed "
               << MAX_PQ_CHUNKS << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::string disk_pq_pivots_path = this->disk_index_file + "_pq_pivots.bin";
    if (file_exists(disk_pq_pivots_path))
    {
        use_disk_index_pq = true;
#ifdef EXEC_ENV_OLS
        // giving 0 chunks to make the pq_table infer from the
        // chunk_offsets file the correct value
        disk_pq_table.load_pq_centroid_bin(files, disk_pq_pivots_path.c_str(), 0);
#else
        // giving 0 chunks to make the pq_table infer from the
        // chunk_offsets file the correct value
        disk_pq_table.load_pq_centroid_bin(disk_pq_pivots_path.c_str(), 0);
#endif
        disk_pq_n_chunks = disk_pq_table.get_num_chunks();
        disk_bytes_per_point = disk_pq_n_chunks * sizeof(_u8); // revising disk_bytes_per_point since DISK PQ is used.
        diskann::cout << "Disk index uses PQ data compressed down to " << disk_pq_n_chunks << " bytes per point."
                      << std::endl;
    }

// read index metadata
#ifdef EXEC_ENV_OLS
    // This is a bit tricky. We have to read the header from the
    // disk_index_file. But  this is now exclusively a preserve of the
    // DiskPriorityIO class. So, we need to estimate how many
    // bytes are needed to store the header and read in that many using our
    // 'standard' aligned file reader approach.
    reader->open(disk_index_file);
    this->setup_thread_data(num_threads);
    this->max_nthreads = num_threads;

    char *bytes = getHeaderBytes();
    ContentBuf buf(bytes, HEADER_SIZE);
    std::basic_istream<char> index_metadata(&buf);
#else
    std::ifstream index_metadata(disk_index_file, std::ios::binary);
#endif

    _u32 nr, nc; // metadata itself is stored as bin format (nr is number of
                 // metadata, nc should be 1)
    READ_U32(index_metadata, nr);
    READ_U32(index_metadata, nc);

    _u64 disk_nnodes;
    _u64 disk_ndims; // can be disk PQ dim if disk_PQ is set to true
    READ_U64(index_metadata, disk_nnodes);
    READ_U64(index_metadata, disk_ndims);

    if (disk_nnodes != num_points)
    {
        diskann::cout << "Mismatch in #points for compressed data file and disk "
                         "index file: "
                      << disk_nnodes << " vs " << num_points << std::endl;
        return -1;
    }

    size_t medoid_id_on_file;
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, max_node_len);
    READ_U64(index_metadata, nnodes_per_sector);
    max_degree = ((max_node_len - disk_bytes_per_point) / sizeof(unsigned)) - 1;

    if (max_degree > MAX_GRAPH_DEGREE)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max graph degree (R) does "
                  "not exceed "
               << MAX_GRAPH_DEGREE << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // setting up concept of frozen points in disk index for streaming-DiskANN
    READ_U64(index_metadata, this->num_frozen_points);
    _u64 file_frozen_id;
    READ_U64(index_metadata, file_frozen_id);
    if (this->num_frozen_points == 1)
        this->frozen_location = file_frozen_id;
    if (this->num_frozen_points == 1)
    {
        diskann::cout << " Detected frozen point in index at location " << this->frozen_location
                      << ". Will not output it at search time." << std::endl;
    }

    READ_U64(index_metadata, this->reorder_data_exists);
    if (this->reorder_data_exists)
    {
        if (this->use_disk_index_pq == false)
        {
            throw ANNException("Reordering is designed for used with disk PQ compression option", -1, __FUNCSIG__,
                               __FILE__, __LINE__);
        }
        READ_U64(index_metadata, this->reorder_data_start_sector);
        READ_U64(index_metadata, this->ndims_reorder_vecs);
        READ_U64(index_metadata, this->nvecs_per_sector);
    }

    diskann::cout << "Disk-Index File Meta-data: ";
    diskann::cout << "# nodes per sector: " << nnodes_per_sector;
    diskann::cout << ", max node len (bytes): " << max_node_len;
    diskann::cout << ", max node degree: " << max_degree << std::endl;

#ifdef EXEC_ENV_OLS
    delete[] bytes;
#else
    index_metadata.close();
#endif

#ifndef EXEC_ENV_OLS
    // open AlignedFileReader handle to index_file
    std::string index_fname(disk_index_file);
    reader->open(index_fname);
    this->setup_thread_data(num_threads);
    this->max_nthreads = num_threads;

#endif

#ifdef EXEC_ENV_OLS
    if (files.fileExists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(files, medoids_file, medoids, num_medoids, tmp_dim);
#else
    if (file_exists(medoids_file))
    {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(medoids_file, medoids, num_medoids, tmp_dim);
#endif

        if (tmp_dim != 1)
        {
            std::stringstream stream;
            stream << "Error loading medoids file. Expected bin format of m times "
                      "1 vector of uint32_t."
                   << std::endl;
            throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
#ifdef EXEC_ENV_OLS
        if (!files.fileExists(centroids_file))
        {
#else
        if (!file_exists(centroids_file))
        {
#endif
            diskann::cout << "Centroid data file not found. Using corresponding vectors "
                             "for the medoids "
                          << std::endl;
            use_medoids_data_as_centroids();
        }
        else
        {
            size_t num_centroids, aligned_tmp_dim;
#ifdef EXEC_ENV_OLS
            diskann::load_aligned_bin<float>(files, centroids_file, centroid_data, num_centroids, tmp_dim,
                                             aligned_tmp_dim);
#else
            diskann::load_aligned_bin<float>(centroids_file, centroid_data, num_centroids, tmp_dim, aligned_tmp_dim);
#endif
            if (aligned_tmp_dim != aligned_dim || num_centroids != num_medoids)
            {
                std::stringstream stream;
                stream << "Error loading centroids data file. Expected bin format of "
                          "m times data_dim vector of float, where m is number of "
                          "medoids "
                          "in medoids file.";
                diskann::cerr << stream.str() << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
    }
    else
    {
        num_medoids = 1;
        medoids = new uint32_t[1];
        medoids[0] = (_u32)(medoid_id_on_file);
        use_medoids_data_as_centroids();
    }

    std::string norm_file = std::string(disk_index_file) + "_max_base_norm.bin";

    if (file_exists(norm_file) && metric == diskann::Metric::INNER_PRODUCT)
    {
        _u64 dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(norm_file, norm_val, dumr, dumc);
        this->max_base_norm = norm_val[0];
        diskann::cout << "Setting re-scaling factor of base vectors to " << this->max_base_norm << std::endl;
        delete[] norm_val;
    }
    diskann::cout << "done.." << std::endl;
    return 0;
}

#ifdef USE_BING_INFRA
bool getNextCompletedRequest(const IOContext &ctx, size_t size, int &completedIndex)
{
    bool waitsRemaining = false;
    long completeCount = ctx.m_completeCount;
    do
    {
        for (int i = 0; i < size; i++)
        {
            auto ithStatus = (*ctx.m_pRequestsStatus)[i];
            if (ithStatus == IOContext::Status::READ_SUCCESS)
            {
                completedIndex = i;
                return true;
            }
            else if (ithStatus == IOContext::Status::READ_WAIT)
            {
                waitsRemaining = true;
            }
        }

        // if we didn't find one in READ_SUCCESS, wait for one to complete.
        if (waitsRemaining)
        {
            WaitOnAddress(&ctx.m_completeCount, &completeCount, sizeof(completeCount), 100);
            // this assumes the knowledge of the reader behavior (implicit
            // contract). need better factoring?
        }
    } while (waitsRemaining);

    completedIndex = -1;
    return false;
}
#endif

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const _u64 k_search, const _u64 l_search,
                                                 _u64 *indices, float *distances, const _u64 beam_width,
                                                 const bool use_reorder_data, QueryStats *stats)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, std::numeric_limits<_u32>::max(),
                       use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const _u64 k_search, const _u64 l_search,
                                                 _u64 *indices, float *distances, const _u64 beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const bool use_reorder_data, QueryStats *stats)
{
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, use_filter, filter_label,
                       std::numeric_limits<_u32>::max(), use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const _u64 k_search, const _u64 l_search,
                                                 _u64 *indices, float *distances, const _u64 beam_width,
                                                 const _u32 io_limit, const bool use_reorder_data, QueryStats *stats)
{
    LabelT dummy_filter = 0;
    cached_beam_search(query1, k_search, l_search, indices, distances, beam_width, false, dummy_filter,
                       std::numeric_limits<_u32>::max(), use_reorder_data, stats);
}

template <typename T, typename LabelT>
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query1, const _u64 k_search, const _u64 l_search,
                                                 _u64 *indices, float *distances, const _u64 beam_width,
                                                 const bool use_filter, const LabelT &filter_label, const _u32 io_limit,
                                                 const bool use_reorder_data, QueryStats *stats)
{
    int32_t filter_num = 0;
    if (use_filter)
    {
        filter_num = get_filter_number(filter_label);
        if (filter_num < 0)
        {
            if (!_use_universal_label)
            {
                return;
            }
            else
            {
                filter_num = _universal_filter_num;
            }
        }
    }

    if (beam_width > MAX_N_SECTOR_READS)
        throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS", -1, __FUNCSIG__, __FILE__, __LINE__);

    ScratchStoreManager<SSDThreadData<T>> manager(this->thread_data);
    auto data = manager.scratch_space();
    IOContext &ctx = data->ctx;
    auto query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->_pq_scratch;

    // reset query scratch
    query_scratch->reset();

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float query_norm = 0;
    T *aligned_query_T = query_scratch->aligned_query_T;
    float *query_float = pq_query_scratch->aligned_query_float;
    float *query_rotated = pq_query_scratch->rotated_query;

    // if inner product, we laso normalize the query and set the last coordinate
    // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
    if (metric == diskann::Metric::INNER_PRODUCT)
    {
        for (size_t i = 0; i < this->data_dim - 1; i++)
        {
            aligned_query_T[i] = query1[i];
            query_norm += query1[i] * query1[i];
        }
        aligned_query_T[this->data_dim - 1] = 0;

        query_norm = std::sqrt(query_norm);

        for (size_t i = 0; i < this->data_dim - 1; i++)
        {
            aligned_query_T[i] /= query_norm;
        }
        pq_query_scratch->set(this->data_dim, aligned_query_T);
    }
    else
    {
        for (size_t i = 0; i < this->data_dim; i++)
        {
            aligned_query_T[i] = query1[i];
        }
        pq_query_scratch->set(this->data_dim, aligned_query_T);
    }

    // pointers to buffers for data
    T *data_buf = query_scratch->coord_scratch;
    _u64 &data_buf_idx = query_scratch->coord_idx;
    _mm_prefetch((char *)data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    // query <-> PQ chunk centers distances
    pq_table.preprocess_query(query_rotated); // center the query and rotate if
                                              // we have a rotation matrix
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query_rotated, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    _u8 *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids, const _u64 n_ids, float *dists_out) {
        diskann::aggregate_coords(ids, n_ids, this->data, this->n_chunks, pq_coord_scratch);
        diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };
    Timer query_timer, io_timer, cpu_timer;

    tsl::robin_set<_u64> &visited = query_scratch->visited;
    NeighborPriorityQueue &retset = query_scratch->retset;
    retset.reserve(l_search);
    std::vector<Neighbor> &full_retset = query_scratch->full_retset;

    _u32 best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();
    if (!use_filter)
    {
        for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++)
        {
            float cur_expanded_dist =
                dist_cmp_float->compare(query_float, centroid_data + aligned_dim * cur_m, (unsigned)aligned_dim);
            if (cur_expanded_dist < best_dist)
            {
                best_medoid = medoids[cur_m];
                best_dist = cur_expanded_dist;
            }
        }
    }
    else if (_filter_to_medoid_id.find(filter_label) != _filter_to_medoid_id.end())
    {
        best_medoid = _filter_to_medoid_id[filter_label];
    }
    else
    {
        throw ANNException("Cannot find medoid for specified filter.", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    compute_dists(&best_medoid, 1, dist_scratch);
    retset.insert(Neighbor(best_medoid, dist_scratch[0]));
    visited.insert(best_medoid);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>> cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    while (retset.has_unexpanded_node() && num_ios < io_limit)
    {
        // clear iteration state
        frontier.clear();
        frontier_nhoods.clear();
        frontier_read_reqs.clear();
        cached_nhoods.clear();
        sector_scratch_idx = 0;
        // find new beam
        _u32 num_seen = 0;
        while (retset.has_unexpanded_node() && frontier.size() < beam_width && num_seen < beam_width)
        {
            auto nbr = retset.closest_unexpanded();
            num_seen++;
            auto iter = nhood_cache.find(nbr.id);
            if (iter != nhood_cache.end())
            {
                cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
                if (stats != nullptr)
                {
                    stats->n_cache_hits++;
                }
            }
            else
            {
                frontier.push_back(nbr.id);
            }
            if (this->count_visited_nodes)
            {
                reinterpret_cast<std::atomic<_u32> &>(this->node_visit_counter[nbr.id].second).fetch_add(1);
            }
        }

        // read nhoods of frontier ids
        if (!frontier.empty())
        {
            if (stats != nullptr)
                stats->n_hops++;
            for (_u64 i = 0; i < frontier.size(); i++)
            {
                auto id = frontier[i];
                std::pair<_u32, char *> fnhood;
                fnhood.first = id;
                fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
                sector_scratch_idx++;
                frontier_nhoods.push_back(fnhood);
                frontier_read_reqs.emplace_back(NODE_SECTOR_NO(((size_t)id)) * SECTOR_LEN, SECTOR_LEN, fnhood.second);
                if (stats != nullptr)
                {
                    stats->n_4k++;
                    stats->n_ios++;
                }
                num_ios++;
            }
            io_timer.reset();
#ifdef USE_BING_INFRA
            reader->read(frontier_read_reqs, ctx, true); // async reader windows.
#else
            reader->read(frontier_read_reqs, ctx); // synchronous IO linux
#endif
            if (stats != nullptr)
            {
                stats->io_us += (double)io_timer.elapsed();
            }
        }

        // process cached nhoods
        for (auto &cached_nhood : cached_nhoods)
        {
            auto global_cache_iter = coord_cache.find(cached_nhood.first);
            T *node_fp_coords_copy = global_cache_iter->second;
            float cur_expanded_dist;
            if (!use_disk_index_pq)
            {
                cur_expanded_dist = dist_cmp->compare(aligned_query_T, node_fp_coords_copy, (unsigned)aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    cur_expanded_dist = disk_pq_table.inner_product(query_float, (_u8 *)node_fp_coords_copy);
                else
                    cur_expanded_dist = disk_pq_table.l2_distance( // disk_pq does not support OPQ yet
                        query_float, (_u8 *)node_fp_coords_copy);
            }
            full_retset.push_back(Neighbor((unsigned)cached_nhood.first, cur_expanded_dist));

            _u64 nnbrs = cached_nhood.second.first;
            unsigned *node_nbrs = cached_nhood.second.second;

            // compute node_nbrs <-> query dists in PQ space
            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (double)nnbrs;
                stats->cpu_us += (double)cpu_timer.elapsed();
            }

            // process prefetched nhood
            for (_u64 m = 0; m < nnbrs; ++m)
            {
                unsigned id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                        continue;

                    if (use_filter && !point_has_label(id, filter_num) && !point_has_label(id, _universal_filter_num))
                        continue;
                    cmps++;
                    float dist = dist_scratch[m];
                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }
        }
#ifdef USE_BING_INFRA
        // process each frontier nhood - compute distances to unvisited nodes
        int completedIndex = -1;
        long requestCount = static_cast<long>(frontier_read_reqs.size());
        // If we issued read requests and if a read is complete or there are reads
        // in wait state, then enter the while loop.
        while (requestCount > 0 && getNextCompletedRequest(ctx, requestCount, completedIndex))
        {
            assert(completedIndex >= 0);
            auto &frontier_nhood = frontier_nhoods[completedIndex];
            (*ctx.m_pRequestsStatus)[completedIndex] = IOContext::PROCESS_COMPLETE;
#else
        for (auto &frontier_nhood : frontier_nhoods)
        {
#endif
            char *node_disk_buf = OFFSET_TO_NODE(frontier_nhood.second, frontier_nhood.first);
            unsigned *node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
            _u64 nnbrs = (_u64)(*node_buf);
            T *node_fp_coords = OFFSET_TO_NODE_COORDS(node_disk_buf);
            //        assert(data_buf_idx < MAX_N_CMPS);
            if (data_buf_idx == MAX_N_CMPS)
                data_buf_idx = 0;

            T *node_fp_coords_copy = data_buf + (data_buf_idx * aligned_dim);
            data_buf_idx++;
            memcpy(node_fp_coords_copy, node_fp_coords, disk_bytes_per_point);
            float cur_expanded_dist;
            if (!use_disk_index_pq)
            {
                cur_expanded_dist = dist_cmp->compare(aligned_query_T, node_fp_coords_copy, (unsigned)aligned_dim);
            }
            else
            {
                if (metric == diskann::Metric::INNER_PRODUCT)
                    cur_expanded_dist = disk_pq_table.inner_product(query_float, (_u8 *)node_fp_coords_copy);
                else
                    cur_expanded_dist = disk_pq_table.l2_distance(query_float, (_u8 *)node_fp_coords_copy);
            }
            full_retset.push_back(Neighbor(frontier_nhood.first, cur_expanded_dist));
            unsigned *node_nbrs = (node_buf + 1);
            // compute node_nbrs <-> query dist in PQ space
            cpu_timer.reset();
            compute_dists(node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (double)nnbrs;
                stats->cpu_us += (double)cpu_timer.elapsed();
            }

            cpu_timer.reset();
            // process prefetch-ed nhood
            for (_u64 m = 0; m < nnbrs; ++m)
            {
                unsigned id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    if (!use_filter && _dummy_pts.find(id) != _dummy_pts.end())
                        continue;

                    if (use_filter && !point_has_label(id, filter_num) && !point_has_label(id, _universal_filter_num))
                        continue;
                    cmps++;
                    float dist = dist_scratch[m];
                    if (stats != nullptr)
                    {
                        stats->n_cmps++;
                    }

                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }

            if (stats != nullptr)
            {
                stats->cpu_us += (double)cpu_timer.elapsed();
            }
        }

        hops++;
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end());

    if (use_reorder_data)
    {
        if (!(this->reorder_data_exists))
        {
            throw ANNException("Requested use of reordering data which does not exist in index "
                               "file",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        std::vector<AlignedRead> vec_read_reqs;

        if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
            full_retset.erase(full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER, full_retset.end());

        for (size_t i = 0; i < full_retset.size(); ++i)
        {
            vec_read_reqs.emplace_back(VECTOR_SECTOR_NO(((size_t)full_retset[i].id)) * SECTOR_LEN, SECTOR_LEN,
                                       sector_scratch + i * SECTOR_LEN);

            if (stats != nullptr)
            {
                stats->n_4k++;
                stats->n_ios++;
            }
        }

        io_timer.reset();
#ifdef USE_BING_INFRA
        reader->read(vec_read_reqs, ctx, false); // sync reader windows.
#else
        reader->read(vec_read_reqs, ctx); // synchronous IO linux
#endif
        if (stats != nullptr)
        {
            stats->io_us += io_timer.elapsed();
        }

        for (size_t i = 0; i < full_retset.size(); ++i)
        {
            auto id = full_retset[i].id;
            auto location = (sector_scratch + i * SECTOR_LEN) + VECTOR_SECTOR_OFFSET(id);
            full_retset[i].distance = dist_cmp->compare(aligned_query_T, (T *)location, this->data_dim);
        }

        std::sort(full_retset.begin(), full_retset.end());
    }

    // copy k_search values
    for (_u64 i = 0; i < k_search; i++)
    {
        indices[i] = full_retset[i].id;

        if (_dummy_pts.find(indices[i]) != _dummy_pts.end())
        {
            indices[i] = _dummy_to_real_map[indices[i]];
        }

        if (distances != nullptr)
        {
            distances[i] = full_retset[i].distance;
            if (metric == diskann::Metric::INNER_PRODUCT)
            {
                // flip the sign to convert min to max
                distances[i] = (-distances[i]);
                // rescale to revert back to original norms (cancelling the effect of
                // base and query pre-processing)
                if (max_base_norm != 0)
                    distances[i] *= (max_base_norm * query_norm);
            }
        }
    }

#ifdef USE_BING_INFRA
    ctx.m_completeCount = 0;
#endif

    if (stats != nullptr)
    {
        stats->total_us = (double)query_timer.elapsed();
    }
}

// range search returns results of all neighbors within distance of range.
// indices and distances need to be pre-allocated of size l_search and the
// return value is the number of matching hits.
template <typename T, typename LabelT>
_u32 PQFlashIndex<T, LabelT>::range_search(const T *query1, const double range, const _u64 min_l_search,
                                           const _u64 max_l_search, std::vector<_u64> &indices,
                                           std::vector<float> &distances, const _u64 min_beam_width, QueryStats *stats)
{
    _u32 res_count = 0;

    bool stop_flag = false;

    _u32 l_search = min_l_search; // starting size of the candidate list
    while (!stop_flag)
    {
        indices.resize(l_search);
        distances.resize(l_search);
        _u64 cur_bw = min_beam_width > (l_search / 5) ? min_beam_width : l_search / 5;
        cur_bw = (cur_bw > 100) ? 100 : cur_bw;
        for (auto &x : distances)
            x = std::numeric_limits<float>::max();
        this->cached_beam_search(query1, l_search, l_search, indices.data(), distances.data(), cur_bw, false, stats);
        for (_u32 i = 0; i < l_search; i++)
        {
            if (distances[i] > (float)range)
            {
                res_count = i;
                break;
            }
            else if (i == l_search - 1)
                res_count = l_search;
        }
        if (res_count < (_u32)(l_search / 2.0))
            stop_flag = true;
        l_search = l_search * 2;
        if (l_search > max_l_search)
            stop_flag = true;
    }
    indices.resize(res_count);
    distances.resize(res_count);
    return res_count;
}

template <typename T, typename LabelT> _u64 PQFlashIndex<T, LabelT>::get_data_dim()
{
    return data_dim;
}

template <typename T, typename LabelT> diskann::Metric PQFlashIndex<T, LabelT>::get_metric()
{
    return this->metric;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename LabelT> char *PQFlashIndex<T, LabelT>::getHeaderBytes()
{
    IOContext &ctx = reader->get_ctx();
    AlignedRead readReq;
    readReq.buf = new char[PQFlashIndex<T, LabelT>::HEADER_SIZE];
    readReq.len = PQFlashIndex<T, LabelT>::HEADER_SIZE;
    readReq.offset = 0;

    std::vector<AlignedRead> readReqs;
    readReqs.push_back(readReq);

    reader->read(readReqs, ctx, false);

    return (char *)readReq.buf;
}
#endif

// instantiations
template class PQFlashIndex<_u8>;
template class PQFlashIndex<_s8>;
template class PQFlashIndex<float>;
template class PQFlashIndex<_u8, uint16_t>;
template class PQFlashIndex<_s8, uint16_t>;
template class PQFlashIndex<float, uint16_t>;

} // namespace diskann
