#include "common_includes.h"
#include <boost/program_options.hpp>
#include <filesystem>
#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"
#include "in_mem_graph_store.h"
#include "dual_graph.h"
#include <signal.h>
#include <iostream>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#include "index.h"
#include "memory_mapper.h"
#include "utils.h"
#include "program_options_utils.hpp"
#include "index_factory.h"
//#include "../apps/utils/compute_groundtruth.cpp"

int train_mode = 0;

void handle_sigterm(int sig)
{
    std::cout << "Received SIGTERM, not terminating." << std::endl;
}


std::vector<uint32_t>
sort_priority_queue(std::priority_queue<std::pair<float, uint32_t>> pq) {
    std::vector<uint32_t> sorted;
    while (!pq.empty()) {
        sorted.push_back(pq.top().second);
        pq.pop();
    }
    std::reverse(sorted.begin(), sorted.end());
    return sorted;
}


template <typename T, typename LabelT = uint32_t>
int same_node_test(std::unique_ptr<diskann::AbstractIndex> index, DualGraph* dual_graph,
                           const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                           const uint32_t recall_at, const std::vector<uint32_t> &Lvec,
                           const std::string &base_file,
                           const bool is_train, const bool use_cached_top1, int topk_num=5, std::string delta_str="0.51",
                           int eval_mode = -1) {

    // dim and num
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;


    // print related
    bool show_qps_per_thread = true;
    bool print_all_recalls = true;
    uint32_t recalls_to_print = 0;
    uint32_t table_width = 0;
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    const uint32_t first_recall = print_all_recalls ? 1 : recall_at;

    // load data, gt, index
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);


    // start query
    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats = std::vector<uint32_t>(query_num, 0);


    std::unordered_map<uint32_t, std::vector<uint32_t>> gt_nn_query_id_map;      // key: nn    id    value: vector of query id
    std::unordered_map<uint32_t, std::vector<uint32_t>> our_nn_query_id_map;      // key: nn    id    value: vector of query id
    std::unordered_map<uint32_t, uint32_t> gt_map;                                  // key: query id    value: gt_nn_id
    std::unordered_map<uint32_t, uint32_t> our_map;                                 // key: query id    value: our_nn_id
    std::unordered_map<uint32_t, std::vector<uint32_t>> route_map;                  // key: query id    value: route


    bool is_calculate_middle = true;
    if (is_train and is_calculate_middle) {
        std::cout << "====Start generated vectors====" << std::endl;

        size_t read_blk_size = 64 * 1024 * 1024;
        cached_ifstream reader(base_file, read_blk_size);
        int npts_i32;
        reader.read((char *)&npts_i32, sizeof(int));
        int test_base_size = (uint32_t)npts_i32;
        float delta = std::stof(delta_str);

        bool calculate_gt = false;

        size_t test_query_num = test_base_size * topk_num;


        diskann::location_t* base_ids = new diskann::location_t[query_aligned_dim * test_query_num];
        diskann::location_t* topk_ids = new diskann::location_t[query_aligned_dim * test_query_num];

        // generate test data
        float* test_query = new float[query_aligned_dim * test_query_num];

        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 100)
        for (int i = 0; i < test_base_size; i++) {
            auto dual_adj_list = dual_graph->get_neighbours(i);
            auto neighbors = index->get_neighbors(i);

            float* base_data = new float[query_aligned_dim];
            index->get_data(base_data, i);

            float* norm_base = new float[query_aligned_dim];
            index->preprocess_query(base_data, query_dim, norm_base);

            {
                std::vector<std::pair<uint32_t, float>> full_adj;
                for (int k = 0; k < dual_adj_list.size(); k++) {
                    uint32_t topk_id = dual_adj_list[k];
                    float distance = index->get_distance(norm_base, topk_id);
                    full_adj.emplace_back(topk_id, distance);
                }
                for (auto neighbor_id : neighbors) {
                    float distance = index->get_distance(norm_base, neighbor_id);
                    full_adj.emplace_back((uint32_t)neighbor_id, distance);
                }

                std::sort(full_adj.begin(), full_adj.end(), [](auto a, auto b) {
                    return a.second < b.second;
                });

//                assert(topk_num - 1  < full_adj.size());

                for (int k = 0; k < topk_num and k < full_adj.size(); k++) {
                    diskann::location_t topk_id = full_adj[k].first;
                    float* topk_data = new float[query_aligned_dim];
                    index->get_data(topk_data, topk_id);

                    float* test_query_start = test_query + (i * topk_num + k) * query_aligned_dim;
                    base_ids[i * topk_num + k] = i;
                    topk_ids[i * topk_num + k] = topk_id;

                    for (int d = 0; d < query_aligned_dim; d++) {
                        test_query_start[d] = delta * base_data[d] + (1 - delta) * topk_data[d];
                    }

                    delete[] topk_data;
                }
            }

            delete[] base_data;
            delete[] norm_base;
        }


        // query
        int shot_base = 0;
        int shot_topk = 0;
        int shot_other = 0;

        int rank_shot_base = 0;
        int rank_shot_topk = 0;
        int rank_shot_other = 0;

        float distance2base_shot_base = 0;
        float distance2base_shot_topk = 0;
        float distance2base_shot_other = 0;

        float avg_distance2base = 0;
        float count_add_edges = 0;
        std::set<std::pair<diskann::location_t, diskann::location_t>> add_edges;


        std::cout << "====Start generated feedback====" << std::endl;
        std::cout << "query size: " << test_base_size * topk_num << std::endl;
        omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic, 100)
        for (int i = 0; i < test_base_size; i++)  {
            if (i % 1000 == 0)
                std::cout << "processed on " << i << " " << std::endl;
            int base_id = i;

            for (int k = 0; k < topk_num; k++) {
                diskann::location_t topk_id = topk_ids[i * topk_num + k];

                float* test_query_start = test_query + (i * topk_num + k) * query_aligned_dim;

                std::vector<uint32_t> test_query_result_ids(recall_at);
                std::vector<float> test_query_result_dists(recall_at);
                std::vector<uint32_t> route;
                int L = Lvec[0];

                index->search_ret_route(
                        i,
                        test_query_start,
                        recall_at,
                        L,
                        test_query_result_ids.data(),
                        route,
                        test_query_result_dists.data()
                        );

                int local_optimum = test_query_result_ids[0];


                if (local_optimum != base_id) {
                        #pragma omp critical
                    {
                        add_edges.insert({local_optimum, base_id});
                    }
                }
            }
        }

        for (auto p : add_edges) {
            dual_graph->add_neighbour(p.first, p.second);
            count_add_edges++;
        }


        {
            std::cout << delta << std::endl;
            std::cout << avg_distance2base / (double) test_query_num << " " << std::endl;
            std::cout << count_add_edges / (double) test_base_size << " " << std::endl;

            std::cout
                    << shot_base / (double) test_query_num << " "
                    << shot_topk / (double) test_query_num << " "
                    << shot_other / (double) test_query_num
                    << std::endl;

            std::cout
                    << rank_shot_base / (double) test_query_num << " "
                    << rank_shot_topk / (double) test_query_num << " "
                    << rank_shot_other / (double) test_query_num
                    << std::endl;

            std::cout
                    << distance2base_shot_base / (double) test_query_num << " "
                    << distance2base_shot_topk / (double) test_query_num << " "
                    << distance2base_shot_other / (double) test_query_num
                    << std::endl;

            std::cout << std::endl;
        }

        delete[] test_query;

    }



    if (print_all_recalls) {
        std::cout << "Using " << num_threads << " threads to search" << std::endl;
        std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
        std::cout.precision(2);
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps"
        << std::setw(20) << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 18 + 20 + 15;
        for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
        {
            std::cout << std::setw(12) << ("Recall@" + std::to_string(curr_recall));
        }
        recalls_to_print = recall_at + 1 - first_recall;
        table_width += recalls_to_print * 12;
        std::cout << std::endl;
        std::cout << std::string(table_width, '=') << std::endl;
    }


    double best_recall = 0.0;
    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
        uint32_t L = Lvec[test_id];

        index->inter_knng_final_count = 0;
        index->inter_aknng_final_count = 0;
        index->inter_knng2_final_count = 0;
        index->add_success = 0;
        index->valid_insert = 0;

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::vector<T *> res = std::vector<T *>();

        std::set<std::pair<diskann::location_t, diskann::location_t>> add_edge_pairs;
        float lid = 0, max_distance = 0;  // computed by 50-th NN


        auto s = std::chrono::high_resolution_clock::now();
        int succ_total = 0, dis_cmp_total = 0;
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 100)
        for (int64_t i = 0; i < (int64_t)query_num; i++) {
            auto qs = std::chrono::high_resolution_clock::now();

            std::vector<uint32_t> route;
            cmp_stats[i] = index->search_ret_route(
                    i,
                    query + i * query_aligned_dim,
                    recall_at,
                    L,
                    query_result_ids[test_id].data() + i * recall_at,
                    route,
                    query_result_dists[test_id].data() + i * recall_at
                    ).second;

            if (use_cached_top1 or is_train) {
                float* norm_query = new float[query_aligned_dim];
                index->preprocess_query(query + i * query_aligned_dim, query_dim, norm_query);


                std::priority_queue<std::pair<float, uint32_t>> result_queue;
                std::unordered_set<uint32_t> result_set;
                bool find_new;
                int succ = 0;
                for (int j = 0; j < recall_at; j++) {
                    result_queue.push({(query_result_dists[test_id].data() + i * recall_at)[j],
                                       (query_result_ids[test_id].data() + i * recall_at)[j]});
                    result_set.insert((query_result_ids[test_id].data() + i * recall_at)[j]);
                }
                for (int j = 0; j < 2; j++) {
                    int dis_cmp = 0;
                    uint32_t top1_id = (query_result_ids[test_id].data() + i * recall_at)[0];

                    if (use_cached_top1) {
                        const auto& neighbors = dual_graph->get_neighbours(top1_id);

//                        int k = (j == 0) ? 10 : 0;
                        for (auto neighbor_id : neighbors) {
                            float distance = index->get_distance(norm_query, neighbor_id);
                            if (distance < result_queue.top().first and result_set.find(neighbor_id) == result_set.end()){
                                result_set.insert(neighbor_id);
                                result_queue.push({distance, neighbor_id});
                                result_queue.pop();
                                succ++;
                            }
                            if (distance < (query_result_dists[test_id].data() + i * recall_at)[0]) {
                                find_new = true;
                            }
                            dis_cmp++;
                            if (dis_cmp > 30)
                                break;
                        }

                    }

//                    #pragma omp critical
//                    {
//                        dis_cmp_total += dis_cmp;
//                    };

                    if (find_new) {
                        continue;
                    } else{
                        break;
                    }
                }
//                #pragma omp critical
//                {
//                    succ_total += succ;
//                };
                delete[] norm_query;

                auto sorted_result = sort_priority_queue(result_queue);

                for (auto j = 0; j < recall_at; j++){
                    query_result_ids[test_id][i * recall_at + j] = sorted_result[j];
                }
            }

            if (is_train) {
                // 注意id没在query里，query只有向量，id应该是tag
                uint32_t *gt_id_vec_start = gt_ids + (uint32_t)gt_dim * i;
                float* gt_dist_vec_start = gt_dists + (uint32_t)gt_dim * i;

                diskann::location_t gt_nn_id = gt_id_vec_start[0];
                diskann::location_t our_nn_id = query_result_ids[test_id][i * recall_at];

                std::vector<diskann::location_t> gt_id_vec(gt_id_vec_start,gt_id_vec_start + (uint32_t)recall_at);

                if (test_id == 0 and our_nn_id != gt_nn_id) {
                    #pragma omp critical
                    {
                        add_edge_pairs.insert({our_nn_id, gt_nn_id});
                    }
                }

            }


            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float)(diff.count() * 1000000);
        }

//        std::cout << 1.0 * succ_total / query_num << " " << 1.0 * dis_cmp_total / query_num << std::endl;

        if (is_train && test_id == 0) {
            for (auto p : add_edge_pairs)
                dual_graph->add_neighbour(p.first, p.second);
            add_edge_pairs.clear();
        }

        if (print_all_recalls) {
            std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

            double displayed_qps = query_num / diff.count();

            if (show_qps_per_thread)
                displayed_qps /= num_threads;

            std::vector<double> recalls;

            recalls.reserve(recalls_to_print);

            uint32_t shot_set[query_num];


            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                            query_result_ids[test_id].data(), recall_at, curr_recall,
                                                            shot_set));
            }

            std::sort(latency_stats.begin(), latency_stats.end());
            double mean_latency =
                    std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

            float avg_cmps = (float)std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) / (float)query_num;

            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(18) << avg_cmps
            << std::setw(20) << (float)mean_latency << std::setw(15)
            << (float)latency_stats[(uint64_t)(0.999 * query_num)];

            for (double recall : recalls)
            {
                std::cout << std::setw(12) << recall;
                best_recall = std::max(recall, best_recall);
            }
            std::cout << std::endl;
        }

    }

}



int main(int argc, char **argv) {
    struct sigaction sa;
    sa.sa_handler = &handle_sigterm;
    sigaction(SIGTERM, &sa, NULL);

#ifdef NDEBUG
    std::cout << "Release Mode" << std::endl;
#else
    std::cout << "Debug Mode" << std::endl;
#endif

    std::string data_type, query_file, gt_file, filter_label, result_path,
    label_type, query_filters_file;
    uint32_t num_threads, K, train_L, topk_num, build_L, build_R, train_R;
    int eval_mode = -1;
    std::vector<uint32_t> Lvec;
    std::vector<std::string> query_filters;
    bool is_train, is_eval, is_validate;


    std::string dataset = "sift1m";
    std::string algo_name = "VAMANA";
    std::string dist_fn = "l2";
    std::string build_A = "1.2";
    std::string delta_str = "0.51";
    K = 10;
    train_L = 50;
    build_L = 100;
    build_R = 32;
    train_R = 5;
    is_train = false;
    is_eval = false;
    is_validate = false;

    if (argc >= 2)
        dataset = std::string(argv[1]);
    if (argc >= 3) {
        train_mode = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        eval_mode = std::stoi(argv[3]);
    }
    if (argc >= 5) {
        is_validate = std::stoi(argv[4]) != 0;
    }
    if (argc >= 6) {
        algo_name = std::string(argv[5]);
    }
    if (argc >= 7) {
        dist_fn = std::string(argv[6]);
    }
    if (argc >= 8) {
        K = std::stoi(argv[7]);
    }
    if (argc >= 9) {
        train_L = std::stoi(argv[8]);
    }
    if (argc >= 10) {
        result_path = std::string(argv[9]);
    }
    if (argc >= 11) {
        build_L = std::stoi(argv[10]);
    }
    if (argc >= 12) {
        build_R = std::stoi(argv[11]);
    }
    if (argc >= 13) {
        build_A = std::string(argv[12]);
    }
    if (argc >= 14) {
        delta_str = std::string(argv[13]);
    }
    if (argc >= 15) {
        train_R = std::stoi(argv[14]);
    }
    topk_num = train_R;
    is_eval = eval_mode != 0;
    is_train = train_mode != 0;
    if (not result_path.empty()) {
        freopen(result_path.c_str(), "w", stdout);
    }


    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                     << std::endl;
        return -1;
    }
    std::string root_dir = "/root/xiaoyao_zhong/";
    std::string data_prefix = root_dir + "dataset/data/" + dataset;
    std::string index_path_prefix = root_dir + "index/";


    std::string index_prefix = algo_name + "/" + algo_name +  "_" + dataset +
            "_learn_R" + std::to_string(build_R) + "_L" + std::to_string(build_L) + "_A" + build_A;
    std::string dg_prefix = "_train_TL" + std::to_string(train_L) + "_TR" + std::to_string(train_R) + "_" + delta_str;

    std::string index_path = index_path_prefix + "index_P/" + index_prefix;
    std::string dual_graph_path, dual_train_graph_path;
    if (train_mode == 1 or train_mode == 3)
        dual_graph_path = index_path_prefix + "index_C/" + index_prefix + dg_prefix + ".dg";
    else
        dual_graph_path = index_path_prefix + "index_empty/" + index_prefix + dg_prefix + ".dg";

    if (train_mode == 1 or eval_mode == 1 or is_validate)
        dual_train_graph_path = index_path_prefix + "index_SC/" + index_prefix + dg_prefix + ".dg";
    else if (train_mode == 2  or eval_mode == 2)
        dual_train_graph_path = index_path_prefix + "index_S/" + index_prefix + dg_prefix + ".dg";
    else if (train_mode == 3  or eval_mode == 3)
        dual_train_graph_path = index_path_prefix + "index_C/" + index_prefix + dg_prefix + ".dg";

    if (is_eval or is_validate) {
        dual_graph_path = dual_train_graph_path;
        std::cout << "Loading Dual Graph from: " + dual_graph_path << std::endl;
        if (not std::filesystem::exists(dual_graph_path)) {
            std::cout << "Error: No Trained index" << std::endl;
            return 0;
        }
    }


    query_file = data_prefix + "/" + dataset + "_gen_query.fbin";
    gt_file = data_prefix + "/" + dataset + "_gen_query_learn_gt100";
    std::string base_file = data_prefix + "/" + dataset + "_learn.fbin";

    if (is_train or is_validate) {
        query_file = data_prefix + "/" + dataset + "_train.fbin";
        gt_file = data_prefix + "/" + dataset + "_train_learn_gt100";
    } else if (is_eval) {
        query_file = data_prefix + "/" + dataset + "_gen_query.fbin";
        gt_file = data_prefix + "/" + dataset + "_gen_query_learn_gt100";
    }

    num_threads = 28;
    if (is_train)
        num_threads = 112;
    if (not is_train) {
        for (int i = 10; i <= 100; i+=10){
            Lvec.push_back(i);
        }
    } else {
        Lvec.assign({train_L});
    }


    float *query, *base = nullptr;
    size_t base_num, base_dim, base_aligned_dim;
    diskann::load_aligned_bin<float>(base_file, base, base_num, base_dim, base_aligned_dim);

    auto config = diskann::IndexConfigBuilder()
            .with_metric(metric)
            .with_dimension(base_dim)
            .with_max_points(0)
            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
            .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
            .with_data_type(diskann_type_to_name<float>())
            .with_label_type(diskann_type_to_name<uint32_t>())
            .with_tag_type(diskann_type_to_name<uint32_t>())
            .build();
    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->load(index_path.c_str(), num_threads, 500);


    DualGraph* dual_graph = new DualGraph(base_num);
    {
        if (std::filesystem::exists(dual_graph_path)) {
            std::cout << "====Start load dual graph====" << std::endl;
            std::cout << "from path: " << dual_graph_path << std::endl;
            dual_graph->load_graph(dual_graph_path, base_num);
        } else if (train_mode == 3) {
            std::cout << "====Start build dual graph====" << std::endl;
            auto s = std::chrono::high_resolution_clock::now();
            omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 100)
            for (int i = 0; i < base_num; i++) {
                if (i % 1000 == 0)
                    std::cout << "processed on " << i << " " << std::endl;

                float *dist = new float[build_R];
                uint32_t *id = new uint32_t[build_R];
                std::vector<uint32_t> route;

                index->search_ret_route(
                        i,
                        base + i * base_aligned_dim,
                        build_R,
                        build_L,
                        id,
                        route,
                        dist
                        );

                std::vector<diskann::location_t> neighbors = index->get_neighbors(i);
                std::unordered_set<diskann::location_t> neighbors_set(neighbors.begin(), neighbors.end());

                for (int j = 0; j < build_R; j++) {
                    if (dual_graph->get_neighbours(i).size() >= 10) {
                        break;
                    }
                    auto cand_id = id[j];
                    if (neighbors_set.find(cand_id) == neighbors_set.end() and cand_id != i) {
                        dual_graph->add_neighbour(i, cand_id);
                    }
                }

                delete[] dist;
                delete[] id;

            }

            std::cout << "Dual Graph Saving to " << dual_graph_path << std::endl;
            dual_graph->save_graph(dual_graph_path, base_num, 0, 0);

            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            std::cout << "Dual Graph Indexing time: " << diff.count() << std::endl;
        }

        if (train_mode == 3) {
            std::cout << "====Exiting====" << dual_graph_path << std::endl;
            return 0;
        }
    }

    if (is_train){
        std::cout << "====Before train====" << std::endl;
        std::cout << "dg path: " << dual_train_graph_path << std::endl;
        std::cout << "dg edges: " << dual_graph->get_num_edges() << std::endl;
    }

    auto s = std::chrono::high_resolution_clock::now();

    same_node_test<float>(std::move(index), dual_graph, query_file, gt_file,
                          num_threads, K, Lvec, base_file,
                          is_train, is_validate or is_eval, topk_num, delta_str, eval_mode);

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Training or Searching time: " << diff.count() << std::endl;

    if (is_train){
        std::cout << "====Dual Graph Saving====" << std::endl;
        std::cout << "dg path: " << dual_train_graph_path << std::endl;
        dual_graph->save_graph(dual_train_graph_path, base_num, 0, 0);
        std::cout << "dg edges: " << dual_graph->get_num_edges() << std::endl;
    }
}
