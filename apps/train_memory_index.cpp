#include "common_includes.h"
#include <boost/program_options.hpp>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"

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



template <typename T, typename LabelT = uint32_t>
int robust_test(diskann::Metric &metric, const std::string &index_path, const std::string &trained_index_path,
                                const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                                const uint32_t recall_at, const std::vector<uint32_t> &Lvec,
                                const bool is_train) {

    // dim and num
    using TagT = uint32_t;
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;


    // print related
    bool show_qps_per_thread = false;
    bool print_all_recalls = true;
    uint32_t recalls_to_print = 0;
    uint32_t table_width = 0;
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    const uint32_t first_recall = print_all_recalls ? 1 : recall_at;


    // load data, gt, index
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);

    auto config = diskann::IndexConfigBuilder()
            .with_metric(metric)
            .with_dimension(query_dim)
            .with_max_points(0)
            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
            .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
            .with_data_type(diskann_type_to_name<T>())
            .with_label_type(diskann_type_to_name<LabelT>())
            .with_tag_type(diskann_type_to_name<TagT>())
            .build();
    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->load(index_path.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));


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


    // start query
    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats = std::vector<uint32_t>(query_num, 0);


    double best_recall = 0.0;
    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
        uint32_t L = Lvec[test_id];

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::vector<T *> res = std::vector<T *>();

        std::vector<std::pair<diskann::location_t, diskann::location_t>> add_edge_pairs;

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++) {
            auto qs = std::chrono::high_resolution_clock::now();

            std::vector<uint32_t> route;

            cmp_stats[i] = index->search_ret_route(
                    query + i * query_aligned_dim,
                    recall_at,
                    L,
                    query_result_ids[test_id].data() + i * recall_at,
                    route,
                    query_result_dists[test_id].data() + i * recall_at
                    ).second;

            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float)(diff.count() * 1000000);
        }

        if (print_all_recalls) {
            std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

            double displayed_qps = query_num / diff.count();

            if (show_qps_per_thread)
                displayed_qps /= num_threads;

            std::vector<double> recalls;

            recalls.reserve(recalls_to_print);
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                            query_result_ids[test_id].data(), recall_at, curr_recall));
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




template <typename T, typename LabelT = uint32_t>
int same_node_test(diskann::Metric &metric, const std::string &index_path, const std::string &trained_index_path,
                                const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                                const uint32_t recall_at, const std::vector<uint32_t> &Lvec,
                                const bool is_train) {

    // dim and num
    using TagT = uint32_t;
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;


    // print related
    bool show_qps_per_thread = false;
    bool print_all_recalls = false;
    uint32_t recalls_to_print = 0;
    uint32_t table_width = 0;
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    const uint32_t first_recall = print_all_recalls ? 1 : recall_at;


    // load data, gt, index
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);

    auto config = diskann::IndexConfigBuilder()
            .with_metric(metric)
            .with_dimension(query_dim)
            .with_max_points(0)
            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
            .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
            .with_data_type(diskann_type_to_name<T>())
            .with_label_type(diskann_type_to_name<LabelT>())
            .with_tag_type(diskann_type_to_name<TagT>())
            .build();
    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->load(index_path.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));


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


    // start query
    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats = std::vector<uint32_t>(query_num, 0);


    std::unordered_map<uint32_t, std::vector<uint32_t>> group_nn_query_id_map;      // key: nn    id    value: vector of query id
    std::unordered_map<uint32_t, uint32_t> gt_map;                                  // key: query id    value: gt_nn_id
    std::unordered_map<uint32_t, uint32_t> our_map;                                 // key: query id    value: our_nn_id
    std::unordered_map<uint32_t, std::vector<uint32_t>> route_map;                  // key: query id    value: route


    double best_recall = 0.0;
    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
        uint32_t L = Lvec[test_id];

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::vector<T *> res = std::vector<T *>();

        std::vector<std::pair<diskann::location_t, diskann::location_t>> add_edge_pairs;
        float lid = 0, max_distance = 0;  // computed by 50-th NN


        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++) {
            auto qs = std::chrono::high_resolution_clock::now();

            std::vector<uint32_t> route;


            cmp_stats[i] = index->search_ret_route(
                    query + i * query_aligned_dim,
                    recall_at,
                    L,
                    query_result_ids[test_id].data() + i * recall_at,
                    route,
                    query_result_dists[test_id].data() + i * recall_at
                    ).second;


            // 注意id没在query里，query只有向量，id应该是tag
            uint32_t *gt_id_vec_start = gt_ids + (uint32_t)gt_dim * i;
            float* gt_dist_vec_start = gt_dists + (uint32_t)gt_dim * i;

            diskann::location_t gt_nn_id = gt_id_vec_start[0];
            diskann::location_t our_nn_id = query_result_ids[test_id][i * recall_at];
            float gt_nn_dist = gt_dist_vec_start[0];
            float our_nn_dist = query_result_dists[test_id][i * recall_at];

            //            std::vector<diskann::location_t> gt_id_vec(gt_id_vec_start,gt_id_vec_start + (uint32_t)recall_at);
            //            std::vector<float> gt_dist_vec(gt_dist_vec_start,gt_dist_vec_start + (uint32_t)recall_at);


            #pragma omp critical
            {
                route_map[i] = route;
                gt_map[i] = gt_nn_id;
                our_map[i] = our_nn_id;

                if (test_id == 0) {
                    group_nn_query_id_map[gt_nn_id].push_back(i);     // todo: 可以替换为our_nn_id
                    lid += *(gt_dist_vec_start + 49);
                    if (*(gt_dist_vec_start + 49) > max_distance) {
                        max_distance = *(gt_dist_vec_start + 49);
                    }
                    //                if (our_nn_id == 9806) {
                    //                    for (int k = 0; k < recall_at; k++)
                    //                        std::cout << query_result_ids[test_id][i * recall_at + k] << " ";
                    //                    std::cout << std::endl;
                    //                    for (int k = 0; k < recall_at; k++)
                    //                        std::cout << query_result_dists[test_id][i * recall_at + k] << " ";
                    //                    std::cout << std::endl << std::endl;
                    //                    our_nn_id = 9806;
                    //                }
                }

                if (is_train && test_id == 0) {
                    if (our_nn_id != gt_nn_id)
                        add_edge_pairs.push_back({our_nn_id, gt_nn_id});
                }

            }

            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float)(diff.count() * 1000000);
        }

        if (is_train && test_id == 0) {
            for (auto p : add_edge_pairs)
                index->add_neighbor(p.first, p.second);
            add_edge_pairs.clear();
            add_edge_pairs.shrink_to_fit();

            index->save(trained_index_path.c_str(), false);
        }

        if (print_all_recalls) {
            std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

            double displayed_qps = query_num / diff.count();

            if (show_qps_per_thread)
                displayed_qps /= num_threads;

            std::vector<double> recalls;

            recalls.reserve(recalls_to_print);
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                            query_result_ids[test_id].data(), recall_at, curr_recall));
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


        /********** lid compute **********/
//        lid /= query_num;
//        lid /= max_distance;
//        std::cout << "lid: " << lid << std::endl;

    }


    /********** num of queries **********/
    size_t test_query_num = 0;
    size_t find_num = 0;
    size_t same_route_count = 0;
    size_t same_gt_nn_count = 0;
    float distance_to_start = 0, distance_to_gt = 0;
    int id = 0, actual_hop = 0;

    for (auto pair : group_nn_query_id_map) {
        auto group_nn_id = pair.first;
        if (pair.second.size() < 2)
            continue;
        else
            same_gt_nn_count += pair.second.size();

        std::unordered_map<uint32_t, size_t> same_nodes;    // key: route_node_id    value: count
        std::vector<size_t> actual_hops;


        /********** find flag check **********/
        bool find_flag = true;
        for (auto query_id : pair.second) {
            test_query_num++;
            std::vector<uint32_t>& route = route_map[query_id];

            for (actual_hop = 0; actual_hop < route.size(); actual_hop++) {
                same_nodes[route[actual_hop]]++;
                if (route[actual_hop] == gt_map[query_id]) {
                    find_num++;
                    break;
                }
            }

            if (actual_hop == route.size())
                find_flag = false;
            actual_hops.push_back(actual_hop);

        }

        /********** same node check **********/
        size_t same_id_size = 0;
        int add_delta = find_flag ? 1 : 0;
        for (auto node : same_nodes) {
            if (node.second == pair.second.size())  // 节点出现次数等于相同query数量
                same_id_size++;
        }

        bool same_flag = true;
        for (auto actual_hop : actual_hops) {
            if (actual_hop + add_delta != same_id_size)
                same_flag = false;

        }
        if (same_flag)
            same_route_count += pair.second.size();



        /********** print route **********/
        bool print_route = true;

        if (not find_flag and print_route) {
            print_route = false;
            std::cout << "gp nn id: " << group_nn_id << std::endl;

            /********** route show **********/
            for (auto query_id : pair.second) {
                std::vector<uint32_t>& route = route_map[query_id];

                std::cout << "gt nn id: " << gt_map[query_id]
                << ", our nn id: " << our_map[query_id]
                << ", query id: " << query_id
                << ", hops: " << route.size()
                << ", actual hops: " << actual_hop
                << ", route: ";

                for (actual_hop = 0; actual_hop < route.size(); actual_hop++) {
                    id = route[actual_hop];
                    std::cout << id << " ";
                    if (id == gt_map[query_id]) {
                        break;
                    }
                }
                std::cout << std::endl;

                /********** route dist show **********/
//                for (actual_hop = 0; actual_hop < route.size(); actual_hop++) {
//                    distance_to_gt = index->get_distance(gt_map[query_id], route[actual_hop]);
//                    std::cout << std::fixed << std::setprecision(2) << distance_to_gt << " ";
//
//                    if (id == gt_map[query_id]) {
//                        break;
//                    }
//                }
//                std::cout << std::endl;
//
//                for (actual_hop = 0; actual_hop < route.size(); actual_hop++) {
//
//                    distance_to_start = index->get_distance(5281, route[actual_hop]);
//                    std::cout << std::fixed << std::setprecision(2) << distance_to_start << " ";
//
//                    if (id == gt_map[query_id]) {
//                        break;
//                    }
//                }
//                std::cout << std::endl;
            }


            /********** same node show **********/
            std::cout << "is same route: " << same_flag << std::endl;
            std::cout << "same size: " << std::setw(2) << same_id_size << ", ids: ";

            for (auto node : route_map[pair.second[0]]) {
                if (same_nodes[node] == pair.second.size())
                    std::cout << std::setw(5) << node << " ";
            }
            std::cout << std::endl;

            std::cout << "same dist to start: ";
            for (auto node : route_map[pair.second[0]]) {
                if (same_nodes[node] == pair.second.size()){
                    distance_to_start = index->get_distance(node, 5281);
                    std::cout << std::setw(5) << std::fixed << std::setprecision(2) << distance_to_start << " ";
                }
            }
            std::cout << std::endl;

            std::cout << "same dist to truth: ";
            for (auto node : route_map[pair.second[0]]) {
                if (same_nodes[node] == pair.second.size()){
                    distance_to_gt = index->get_distance(node, gt_map[pair.second[0]]);
                    std::cout << std::setw(5) << std::fixed << std::setprecision(2) << distance_to_gt << " ";
                }
            }
            std::cout << std::endl;

            std::cout << std::endl;
        }


    }

    if (not print_all_recalls)
        std::cout
        << test_query_num << " "
        << find_num << " "
        << same_gt_nn_count << " "
        << same_route_count << " "
        << std::endl;

}





template <typename T, typename LabelT = uint32_t>
int search_memory_index(diskann::Metric &metric, const std::string &index_path, const std::string &trained_index_path,
                                const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                                const uint32_t recall_at, const std::vector<uint32_t> &Lvec,
                                const bool is_train) {

    // dim and num
    using TagT = uint32_t;
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;


    // print related
    bool show_qps_per_thread = false;
    bool print_all_recalls = true;
    uint32_t recalls_to_print = 0;
    uint32_t table_width = 0;
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    const uint32_t first_recall = print_all_recalls ? 1 : recall_at;


    // load data, gt, index
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);

    auto config = diskann::IndexConfigBuilder()
            .with_metric(metric)
            .with_dimension(query_dim)
            .with_max_points(0)
            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
            .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
            .with_data_type(diskann_type_to_name<T>())
            .with_label_type(diskann_type_to_name<LabelT>())
            .with_tag_type(diskann_type_to_name<TagT>())
            .build();
    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->load(index_path.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));


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


    // start query
    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats = std::vector<uint32_t>(query_num, 0);


    double best_recall = 0.0;
    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
        uint32_t L = Lvec[test_id];

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::vector<T *> res = std::vector<T *>();

        std::vector<std::pair<diskann::location_t, diskann::location_t>> add_edge_pairs;

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++) {
            auto qs = std::chrono::high_resolution_clock::now();

            std::vector<uint32_t> route;

            cmp_stats[i] = index->search_ret_route(
                    query + i * query_aligned_dim,
                    recall_at,
                    L,
                    query_result_ids[test_id].data() + i * recall_at,
                    route,
                    query_result_dists[test_id].data() + i * recall_at
                    ).second;

            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float)(diff.count() * 1000000);
        }

        if (print_all_recalls) {
            std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

            double displayed_qps = query_num / diff.count();

            if (show_qps_per_thread)
                displayed_qps /= num_threads;

            std::vector<double> recalls;

            recalls.reserve(recalls_to_print);
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                            query_result_ids[test_id].data(), recall_at, curr_recall));
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







int main() {
    std::string data_type, dist_fn, index_path_prefix, trained_index_path_prefix, query_file, gt_file, filter_label,
    label_type, query_filters_file;
    uint32_t num_threads, K;
    std::vector<uint32_t> Lvec;
    std::vector<std::string> query_filters;
    bool is_train, is_eval;

    std::string dataset = "gist";
    is_train = false;
    is_eval = true;

    std::string data_prefix = "data/" + dataset;
    index_path_prefix = data_prefix + "/index_" + dataset + "_learn_R32_L50_A1.2";
    trained_index_path_prefix = data_prefix + "/index_" + dataset + "_train_R32_L50_A1.2";

    if (is_eval) {
        index_path_prefix = trained_index_path_prefix;
        is_train = false;
    }

    dataset = "gist_random";
    query_file = data_prefix + "/" + dataset + "_query.fbin";
    gt_file = data_prefix + "/" + dataset + "_query_learn_gt100";


    data_type = "float";
    dist_fn = "l2";
    num_threads = 20;
    K = 10;
    Lvec.assign({50});
    if (not is_train)
        Lvec.assign({10, 20, 30, 40, 50, 100});

    Lvec.assign({50});
//    Lvec.assign({10, 20, 30, 40, 50, 100});
    diskann::Metric metric = diskann::Metric::L2;;

    same_node_test<float>(metric, index_path_prefix, trained_index_path_prefix, query_file, gt_file,
                               num_threads, K, Lvec,
                               is_train);

















//    const size_t query_dim = 128;
//    const size_t query_num = 10000;
//    const size_t num_frozen_pts = 0;
//    auto search_param = diskann::IndexSearchParams(100, 1)
//    auto config = diskann::IndexConfigBuilder()
//            .with_metric(metric)
//            .with_dimension(query_dim)
//            .with_max_points(query_num)
//            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
//            .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
//            .with_data_type(diskann_type_to_name<float>())
//            .with_label_type(diskann_type_to_name<uint32_t>())
//            .with_tag_type(diskann_type_to_name<uint32_t>())
//            .with_index_search_params(search_param)
//            .build();
//
//    auto index_factory = diskann::IndexFactory(config);
//    auto index = index_factory.create_instance();
//
//
//    float* query = new float[query_dim * query_num];
//
//    std::default_random_engine generator(std::time(nullptr));
//    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
//
//    for (size_t i = 0; i < query_num; ++i) {
//        float random_float = distribution(generator);
//        query[i] = random_float;
//    }
//    std::vector<uint32_t> tags;
//    for (int i = 0; i < 10000; ++i) {
//        tags.push_back(i);
//    }
//    index->build(query, 10000, tags);
//    uint32_t ids;
//    float distance;
//
//    index->search(query + 9 * query_dim, 1, 200, &ids, &distance);
//
//    std::cout << ids;
}