
#ifndef HNSWLIB_DUAL_GRAPH_H
#define HNSWLIB_DUAL_GRAPH_H

//#include "util.h"

class DualGraph {
public:
    DualGraph(size_t build_size) {
        _graph.resize(build_size);
    }

    int save_graph(const std::string &filename, const size_t num_points,
                   const size_t num_frozen_points, const uint32_t start) {
        delete_file(filename);
        std::ofstream out;
        open_file_to_write(out, filename);

        size_t file_offset = 0;
        out.seekp(file_offset, out.beg);
        size_t index_size = 24;
        uint32_t max_degree = 0;
        out.write((char *)&index_size, sizeof(uint64_t));
        out.write((char *)&_max_observed_degree, sizeof(uint32_t));
        uint32_t ep_u32 = start;
        out.write((char *)&ep_u32, sizeof(uint32_t));
        out.write((char *)&num_frozen_points, sizeof(size_t));

        // Note: num_points = _nd + _num_frozen_points
        for (uint32_t i = 0; i < num_points; i++)
        {
            uint32_t GK = (uint32_t)_graph[i].size();
            out.write((char *)&GK, sizeof(uint32_t));
            out.write((char *)_graph[i].data(), GK * sizeof(uint32_t));
            max_degree = _graph[i].size() > max_degree ? (uint32_t)_graph[i].size() : max_degree;
            index_size += (size_t)(sizeof(uint32_t) * (GK + 1));
        }
        out.seekp(file_offset, out.beg);
        out.write((char *)&index_size, sizeof(uint64_t));
        out.write((char *)&max_degree, sizeof(uint32_t));
        out.close();
        return (int)index_size;
    }


    std::tuple<uint32_t, uint32_t, size_t> load_graph(const std::string &filename,
                                                      size_t expected_num_points) {
        size_t expected_file_size;
        size_t file_frozen_pts;
        uint32_t start;
        size_t file_offset = 0; // will need this for single file format support

        std::ifstream in;
        in.exceptions(std::ios::badbit | std::ios::failbit);
        in.open(filename, std::ios::binary);
        in.seekg(file_offset, in.beg);
        in.read((char *)&expected_file_size, sizeof(size_t));
        in.read((char *)&_max_observed_degree, sizeof(uint32_t));
        in.read((char *)&start, sizeof(uint32_t));
        in.read((char *)&file_frozen_pts, sizeof(size_t));
        size_t vamana_metadata_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

        std::cout << "Loading graph " << filename << "..." << std::flush;

        _graph.resize(expected_num_points);

        size_t bytes_read = vamana_metadata_size;
        size_t cc = 0;
        uint32_t nodes_read = 0;
        while (bytes_read != expected_file_size)
        {
            uint32_t k;
            in.read((char *)&k, sizeof(uint32_t));

            cc += k;
            ++nodes_read;
            std::vector<uint32_t> tmp(k);
            tmp.reserve(k);
            in.read((char *)tmp.data(), k * sizeof(uint32_t));
            _graph[nodes_read - 1].swap(tmp);
            bytes_read += sizeof(uint32_t) * ((size_t)k + 1);
            if (nodes_read % 10000000 == 0)
                std::cout << "." << std::flush;
        }

        std::cout << "done. Index has " << nodes_read << " nodes and " << cc << " out-edges, _start is set to " << start
        << std::endl;
        return std::make_tuple(nodes_read, start, file_frozen_pts);
    }

    const std::vector<uint32_t> &get_neighbours(const uint32_t i) const
    {
        if (i >= _graph.size()) {
            return {};
        }
        return _graph.at(i);
    }

    void add_neighbour(const uint32_t i, uint32_t neighbour_id)
    {
        for (auto id : _graph[i]) {
            if (id == neighbour_id) {
                return ;
            }
        }
        _graph[i].emplace_back(neighbour_id);
        if (_max_observed_degree < _graph[i].size())
        {
            _max_observed_degree = (uint32_t)(_graph[i].size());
        }
    }

    uint32_t get_num_edges() const {
        uint32_t ret = 0;
        for (const auto& edges : _graph) {
            ret += edges.size();
        }
        return ret;
    }


private:
    size_t _max_range_of_graph = 0;
    uint32_t _max_observed_degree = 0;

    std::vector<std::vector<uint32_t>> _graph;
};

#endif //HNSWLIB_DUAL_GRAPH_H
