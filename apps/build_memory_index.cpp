// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"
#include "program_options_utils.hpp"
#include <signal.h>
#include <iostream>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

#include "memory_mapper.h"
#include "ann_exception.h"
#include "index_factory.h"

namespace po = boost::program_options;

void handle_sigterm(int sig)
{
    std::cout << "Received SIGTERM, not terminating." << std::endl;
}

int main(int argc, char **argv)
{
    struct sigaction sa;
    sa.sa_handler = &handle_sigterm;
    sigaction(SIGTERM, &sa, NULL);

    std::string data_type, dist_fn, data_path, index_path_prefix, label_file, universal_label, label_type;
    std::string algo_name, dataset_name;
    uint32_t num_threads, R, L, Lf, build_PQ_bytes;
    float alpha;
    bool use_pq_build, use_opq;

    po::options_description desc{
        program_options_utils::make_program_description("build_memory_index", "Build a memory-based DiskANN index.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                                       program_options_utils::INPUT_DATA_PATH);
        required_configs.add_options()("algorithm", po::value<std::string>(&algo_name)->required(),
                program_options_utils::INPUT_ALGORITHM_NAME);
        required_configs.add_options()("dataset", po::value<std::string>(&dataset_name)->required(),
                program_options_utils::INPUT_DATASET_NAME);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE);
        optional_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);
        optional_configs.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                                       program_options_utils::GRAPH_BUILD_ALPHA);
        optional_configs.add_options()("build_PQ_bytes", po::value<uint32_t>(&build_PQ_bytes)->default_value(0),
                                       program_options_utils::BUIlD_GRAPH_PQ_BYTES);
        optional_configs.add_options()("use_opq", po::bool_switch()->default_value(false),
                                       program_options_utils::USE_OPQ);
        optional_configs.add_options()("label_file", po::value<std::string>(&label_file)->default_value(""),
                                       program_options_utils::LABEL_FILE);
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       program_options_utils::UNIVERSAL_LABEL);

        optional_configs.add_options()("FilteredLbuild", po::value<uint32_t>(&Lf)->default_value(0),
                                       program_options_utils::FILTERED_LBUILD);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        use_pq_build = (build_PQ_bytes > 0);
        use_opq = vm["use_opq"].as<bool>();
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
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

    try
    {
        diskann::cout << "Starting index build with R: " << R << "  Lbuild: " << L << "  alpha: " << alpha
                      << "  #threads: " << num_threads << std::endl;

        size_t data_num, data_dim;
        diskann::get_bin_metadata(data_path, data_num, data_dim);

        //        size_t data_num = 10000, data_dim = 128;

        auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                      .with_filter_list_size(Lf)
                                      .with_alpha(alpha)
                                      .with_saturate_graph(false)
                                      .with_num_threads(num_threads)
                                      .build();

        auto filter_params = diskann::IndexFilterParamsBuilder()
                                 .with_universal_label(universal_label)
                                 .with_label_file(label_file)
                                 .with_save_path_prefix(index_path_prefix)
                                 .build();
        auto config = diskann::IndexConfigBuilder()
                          .with_metric(metric)
                          .with_dimension(data_dim)
                          .with_max_points(data_num)
                          .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                          .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                          .with_data_type(data_type)
                          .with_label_type(label_type)
                          .is_dynamic_index(false)
                          .with_index_write_params(index_build_params)
                          .is_enable_tags(false)
                          .is_use_opq(use_opq)
                          .is_pq_dist_build(use_pq_build)
                          .with_num_pq_chunks(build_PQ_bytes)
                          .build();

//        float* query = new float[data_dim * data_num];
//
//        std::default_random_engine generator(std::time(nullptr));
//        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
//
//        for (size_t i = 0; i < data_dim * data_num; ++i) {
//            float random_float = distribution(generator);
//            query[i] = random_float;
//        }
//        std::vector<uint32_t> tags;
//        for (int i = 0; i < 10000; ++i) {
//            tags.push_back(i);
//        }

        auto index_factory = diskann::IndexFactory(config);
        auto index = index_factory.create_instance();
        index->use_cached_top1 = false;
        index->use_knn_graph = false;


        std::unordered_map<std::string, float> tau_map = {
                {"sift1m", 10},
                {"gist1m", 0.04},
                {"sift-128-euclidean", 10},
                {"gist-960-euclidean", 0.04},
                {"deep-image-96-angular", 0.04},
                {"glove-100-angular", 0.2},
                {"glove-200-angular", 0.4},

                {"fashion-mnist-784-euclidean", 0.4},
                {"kosarak-jaccard", 0.4},
                {"mnist-784-euclidean", 0.4}
        };


        if (algo_name == "TAUMNG") {
            index->tau = tau_map[dataset_name];
            index->strategy = diskann::AbstractIndex::TAUMNG;

        }
        if (algo_name == "VAMANA") {
            index->strategy = diskann::AbstractIndex::VAMANA;
        }
        if (algo_name == "NSG") {
            index->strategy = diskann::AbstractIndex::NSG;
        }


        index->build(data_path, data_num, filter_params);
//        index->build(query, data_num, tags);
        index->save(index_path_prefix.c_str());
        index.reset();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index build failed." << std::endl;
        return -1;
    }

}
// build
// --data_type float --dist_fn l2 --data_path data/sift/sift_learn.fbin --index_path_prefix data/sift/index_sift_learn_R32_L50_A1.2 -R 32 -L 50 --alpha 1.2
// --data_type float --dist_fn l2 --data_path data/random/random_learn.fbin --index_path_prefix data/random/index_random_learn_R32_L50_A1.2 -R 32 -L 50 --alpha 1.2
// --data_type float --dist_fn l2 --data_path data/gist/gist_learn.fbin --index_path_prefix data/gist/index_gist_learn_R32_L50_A1.2 -R 32 -L 50 --alpha 1.2
// --data_type float --dist_fn cosine --data_path data/gist/gist_learn.fbin --index_path_prefix data/gist/index_gist_learn_R32_L50_A1.2_cosine -R 32 -L 50 --alpha 1.2
// --data_type float --dist_fn l2 --data_path data/gist_random/gist_random_learn.fbin --index_path_prefix data/gist_random/index_gist_random_learn_R32_L50_A1.2 -R 32 -L 50 --alpha 1.2

// --data_type float --dist_fn l2 --data_path data/test/test_learn.fbin --index_path_prefix data/test/index_test_learn_R32_L50_A1.2 -R 32 -L 50 --alpha 1.2



// compute gt
// --data_type float --dist_fn l2 --base_file data/sift/sift_learn.fbin --query_file  data/sift/sift_query.fbin --gt_file data/sift/sift_query_learn_gt100 --K 100
// --data_type float --dist_fn l2 --base_file data/sift/sift_learn.fbin --query_file  data/sift/sift_train.fbin --gt_file data/sift/sift_train_learn_gt100 --K 100


// --data_type float --dist_fn l2 --base_file data/random/random_learn.fbin --query_file  data/random/random_query.fbin --gt_file data/random/random_query_learn_gt100 --K 100

// --data_type float --dist_fn l2 --base_file data/gist/gist_learn.fbin --query_file  data/gist/gist_query.fbin --gt_file data/gist/gist_query_learn_gt100 --K 100
// --data_type float --dist_fn l2 --base_file data/gist/gist_learn.fbin --query_file  data/gist/gist_random_query.fbin --gt_file data/gist/gist_random_query_learn_gt100 --K 100
// --data_type float --dist_fn l2 --base_file data/gist/gist_learn.fbin --query_file  data/gist/gist_random_train.fbin --gt_file data/gist/gist_random_train_learn_gt100 --K 100

// --data_type float --dist_fn cosine --base_file data/gist/gist_learn.fbin --query_file  data/gist/gist_query.fbin --gt_file data/gist/gist_query_learn_gt100_cosine --K 100
// --data_type float --dist_fn cosine --base_file data/gist/gist_learn.fbin --query_file  data/gist/gist_random_query.fbin --gt_file data/gist/gist_random_query_learn_gt100_cosine --K 100
// --data_type float --dist_fn cosine --base_file data/gist/gist_learn.fbin --query_file  data/gist/gist_random_train.fbin --gt_file data/gist/gist_random_train_learn_gt100_cosine --K 100

// --data_type float --dist_fn l2 --base_file data/test/test_learn.fbin --query_file  data/test/test_query_train.fbin --gt_file data/test/test_query_train_learn_gt100 --K 100
// --data_type float --dist_fn l2 --base_file data/test/test_learn.fbin --query_file  data/test/test_query.fbin --gt_file data/test/test_query_learn_gt100 --K 100

// --data_type float --dist_fn l2 --base_file data/gist_random/gist_random_learn.fbin --query_file  data/gist_random/gist_random_learn.fbin --gt_file data/gist_random/gist_random_learn_learn_gt100.fbin --K 100