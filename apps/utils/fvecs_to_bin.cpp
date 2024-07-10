// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"
#include <signal.h>
void handle_sigterm(int sig)
{
    std::cout << "Received SIGTERM, not terminating." << std::endl;
}

// Convert float types
void block_convert_float(std::ifstream &reader, std::ofstream &writer, float *read_buf, float *write_buf,
                         size_t npts, size_t ndims, size_t jump_npts)
{
    reader.read((char *)read_buf, (npts + jump_npts) * (ndims * sizeof(float) + sizeof(uint32_t)));
    for (size_t i = 0; i < npts; i++)
    {
//        std::vector<float> vec((read_buf + (i + jump_npts) * (ndims + 1)) + 1, (read_buf + (i + 1 + jump_npts) * (ndims + 1)) + 1);

        memcpy(write_buf + i * ndims, (read_buf + (i + jump_npts) * (ndims + 1)) + 1, ndims * sizeof(float));
        // memcpy(write_buf[i * ndims], read_
        // read_buf 里面存了1位uint和128位向量数buf[i * (ndims + 1) + 1], ndims * sizeof(float));
        // write_buf 里面只保存128维的向量数据, writer要先写npts和ndims
    }
    writer.write((char *)write_buf, npts * ndims * sizeof(float));
}

// Convert byte types
void block_convert_byte(std::ifstream &reader, std::ofstream &writer, uint8_t *read_buf, uint8_t *write_buf,
                        size_t npts, size_t ndims)
{
    reader.read((char *)read_buf, npts * (ndims * sizeof(uint8_t) + sizeof(uint32_t)));
    for (size_t i = 0; i < npts; i++)
    {
        memcpy(write_buf + i * ndims, (read_buf + i * (ndims + sizeof(uint32_t))) + sizeof(uint32_t),
               ndims * sizeof(uint8_t));
    }
    writer.write((char *)write_buf, npts * ndims * sizeof(uint8_t));
}


void generate_random_dataset_fbin() {
    std::string random_data_bin_dir = "/app/DiskANN/build/data/random";
    std::string random_learn_path = random_data_bin_dir + "/random_learn.fbin";
    std::string random_query_path = random_data_bin_dir + "/random_query.fbin";
    std::string random_train_path = random_data_bin_dir + "/random_train.fbin";

    std::ofstream learn_writer(random_learn_path, std::ios::binary);
    std::ofstream query_writer(random_query_path, std::ios::binary);
    std::ofstream train_writer(random_train_path, std::ios::binary);

    uint32_t chunk = 1;
    uint32_t data_dim = 256;
    size_t learn_data_size = 100000;
    size_t query_data_size = 10000;
    size_t train_data_size = 10000;

    float *learn_write_buf = new float[chunk * data_dim * learn_data_size];
    float *query_write_buf = new float[chunk * data_dim * query_data_size];
    float *train_write_buf = new float[chunk * data_dim * train_data_size];

    std::default_random_engine generator(std::time(nullptr));
    std::uniform_real_distribution<float> distribution(0.0f, 10.0f);

    for (int i = 0; i < learn_data_size; ++i) {
        for (int j = 0; j < data_dim; ++j) {
            learn_write_buf[i * data_dim + j] = distribution(generator);
        }
    }

    for (int i = 0; i < query_data_size; ++i) {
        for (int j = 0; j < data_dim; ++j) {
            query_write_buf[i * data_dim + j] = distribution(generator);
        }
    }

    for (int i = 0; i < train_data_size; ++i) {
        for (int j = 0; j < data_dim; ++j) {
            train_write_buf[i * data_dim + j] = distribution(generator);
        }
    }

    learn_writer.write((char *)&learn_data_size, sizeof(int32_t));
    learn_writer.write((char *)&data_dim, sizeof(int32_t));
    learn_writer.write((char *)learn_write_buf, learn_data_size * data_dim * sizeof(float));

    query_writer.write((char *)&query_data_size, sizeof(int32_t));
    query_writer.write((char *)&data_dim, sizeof(int32_t));
    query_writer.write((char *)query_write_buf, query_data_size * data_dim * sizeof(float));

    train_writer.write((char *)&train_data_size, sizeof(int32_t));
    train_writer.write((char *)&data_dim, sizeof(int32_t));
    train_writer.write((char *)train_write_buf, train_data_size * data_dim * sizeof(float));

    delete[] learn_write_buf;
    delete[] query_write_buf;
    delete[] train_write_buf;

    train_writer.close();
    query_writer.close();
    learn_writer.close();

    return ;
}


void generate_random_data_based_on_origin(std::string base_data_path, std::string generated_data_path,
                                          size_t num_base = 10000,
                                          size_t num_copy = 1, size_t jump_npts = 0, int sampled_base_num = 50000) {

    std::ifstream reader(base_data_path, std::ios::binary | std::ios::ate);
    std::ofstream writer(generated_data_path, std::ios::binary);

    size_t fsize = reader.tellg();
    reader.seekg(0, std::ios::beg);
    uint32_t ndims_u32;
    reader.read((char *)&ndims_u32, sizeof(uint32_t));
    reader.seekg(0, std::ios::beg);

    int datasize = sizeof(float);
    size_t ndims = (size_t)ndims_u32;
    size_t npts = fsize / ((ndims * datasize) + sizeof(uint32_t));
    npts = std::min(npts, (size_t)1000000);

    uint8_t *read_buf = new uint8_t[npts * ((ndims * datasize) + sizeof(uint32_t))];
    reader.read((char *)read_buf, (npts) * (ndims * sizeof(float) + sizeof(uint32_t)));


    size_t npts_generated = num_copy * num_base;
    writer.write((char *)&npts_generated, sizeof(int32_t));
    writer.write((char *)&ndims, sizeof(int32_t));

    float *write_buf = new float[npts_generated * ndims * datasize];

    float avg_dim = 0;
    float base_avg_distance = 0;
    float avg_distance_per_dim = 0;
    int count_pair = 0;
    for (int i = 0; i < sampled_base_num; i++) {
        float* query = ((float*)read_buf + (i) * (ndims + 1)) + 1;

        float tmp = 0;
        for (int dim = 0; dim < ndims; dim++) {
            tmp += *(query + dim);
        }
        avg_dim += (tmp / (float)ndims);


        for (int j = 0; j < 1000; j++) {
            if (i == j)
                continue;
            count_pair++;
            float* base = ((float*)read_buf + (j) * (ndims + 1)) + 1;

            float distance = 0;
            for (int dim = 0; dim < ndims; dim++) {
                distance += (query[dim] - base[dim]) * (query[dim] - base[dim]);
            }
            base_avg_distance += distance;
        }

        for (int j = 0; j < 1000; j++) {
            if (i == j)
                continue;
            count_pair++;
            float* base = ((float*)read_buf + (j) * (ndims + 1)) + 1;

            float distance_per_dim = 0;
            for (int dim = 0; dim < ndims; dim++) {
                distance_per_dim += std::fabs(query[dim] - base[dim]);
            }
            avg_distance_per_dim += (distance_per_dim / (1.0 * ndims));
        }
    }
    avg_dim /= sampled_base_num;
    base_avg_distance /= (1.0 * count_pair);
    avg_distance_per_dim /= (1.0 * count_pair);


    avg_dim /= 2.5;



    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1 * avg_dim, avg_dim);

    num_base = 1;
    num_copy = 10000;
    for (int i = 0; i < num_base; i++) {
        float* query = ((float*)read_buf + (i + jump_npts) * (ndims + 1)) + 1;
        float data;

        for (int j = 0; j < num_copy; j++) {

            for (int dim = 0; dim < ndims; dim++) {
                data = *(query + dim) + dis(gen);
                *(write_buf + (i * num_copy + j) * ndims + dim) = data;
            }

        }

    }

    float avg_distance_to_base = 0;
    float min_distance = 10000;
    float avg_distance = 0;
    float max_distance = 0;


    float min_distance_to_query_per_dim = 10000;
    float avg_distance_to_base_per_dim = 0;
    float avg_distance_to_query_per_dim = 0;
    float max_distance_to_query_per_dim = 0;

    count_pair = 0;

    float* base = ((float*)read_buf + (0 + jump_npts) * (ndims + 1)) + 1;

    for (int i = 0; i < num_copy; i++) {
        float* query_i = write_buf + i * ndims;

        float distance_to_base = 0;
        float distance_to_base_per_dim = 0;
        for (int dim = 0; dim < ndims; dim++) {
            distance_to_base += (query_i[dim] - base[dim]) * (query_i[dim] - base[dim]);
            distance_to_base_per_dim +=  std::fabs(query_i[dim] - base[dim]);
        }
        avg_distance_to_base += distance_to_base;
        avg_distance_to_base_per_dim += (distance_to_base_per_dim / (1.0 * ndims));

        for (int j = 0; j < num_copy; j++) {
            if (i == j)
                continue;
            count_pair++;

            float* query_j = write_buf + j * ndims;

            float distance = 0;
            for (int dim = 0; dim < ndims; dim++) {
                distance += (query_i[dim] - query_j[dim]) * (query_i[dim] - query_j[dim]);
            }

            if (distance > max_distance)
                max_distance = distance;
            if (distance < min_distance)
                min_distance = distance;
            avg_distance += distance;
        }

        for (int j = 0; j < num_copy; j++) {
            if (i == j)
                continue;
            count_pair++;

            float* query_j = write_buf + j * ndims;

            float distance = 0;
            for (int dim = 0; dim < ndims; dim++) {
                auto tmp_distance =  std::fabs(query_i[dim] - query_j[dim]);
                distance += tmp_distance;

                if (tmp_distance > max_distance_to_query_per_dim)
                    max_distance_to_query_per_dim = tmp_distance;
                if (tmp_distance < min_distance_to_query_per_dim) {

                    min_distance_to_query_per_dim = tmp_distance;
                    if (tmp_distance == 0 or tmp_distance < 0.000001)
                        std::cout << query_i[dim] << " " << query_j[dim] << std::endl;
                }
            }
            avg_distance_to_query_per_dim += (distance / (1.0 * ndims));
        }
    }
    avg_distance_to_base_per_dim /= (1.0 * num_copy);
    avg_distance /= (1.0 * count_pair);
    avg_distance_to_query_per_dim /= (1.0 * count_pair);
    avg_distance_to_base /= (1.0 * num_copy);


    std::cout << avg_dim * 2.5 << " " << avg_dim  <<std::endl;
    std::cout << avg_distance_per_dim << std::endl;
    std::cout << avg_distance_to_base_per_dim << std::endl;
    std::cout << min_distance_to_query_per_dim << std::endl;
    std::cout << avg_distance_to_query_per_dim << std::endl;
    std::cout << max_distance_to_query_per_dim << std::endl;
    std::cout << std::endl;


    std::cout << base_avg_distance << std::endl;
    std::cout << avg_distance_to_base << std::endl;
    std::cout << min_distance << std::endl;
    std::cout << avg_distance << std::endl;
    std::cout << max_distance << std::endl;

    delete[] write_buf;
    delete[] read_buf;

    writer.close();
    reader.close();
    exit(0);


    writer.write((char *)write_buf, npts_generated * ndims * sizeof(float));

    delete[] write_buf;
    delete[] read_buf;

    writer.close();
    reader.close();
}



int main(int argc, char **argv)
{
    struct sigaction sa;
    sa.sa_handler = &handle_sigterm;
    sigaction(SIGTERM, &sa, NULL);

//    query_file = data_dir + "/" + dataset + "/" + dataset + "_query.fbin";
//    train_file = data_dir + "/" + dataset + "/" + dataset + "_train.fbin";
//    base_file = data_dir + "/" + dataset + "/" + dataset + "_base.fvecs";
//    return 0;
//
//    generate_random_dataset_fbin();
//    return 0;

    size_t target_npts = 0;
    size_t jump_npts = 0;
    int is_generate = 0;
    size_t generate_npts = 10000;

    if (argc < 4)
    {
        std::cout << argv[0] << " <float/int8/uint8> input_vecs output_bin" << std::endl;
        exit(-1);
    }
    if (argc >= 5)
        target_npts = std::stoi(argv[4]);
    if (argc >= 6)
        jump_npts = std::stoi(argv[5]);
    if (argc >= 7)
        is_generate = std::stoi(argv[6]);
    if (argc >= 8)
        generate_npts = std::stoi(argv[7]);


    int datasize = sizeof(float);

    if (strcmp(argv[1], "uint8") == 0 || strcmp(argv[1], "int8") == 0)
    {
        datasize = sizeof(uint8_t);
    }
    else if (strcmp(argv[1], "float") != 0)
    {
        std::cout << "Error: type not supported. Use float/int8/uint8" << std::endl;
        exit(-1);
    }

    std::ifstream reader(argv[2], std::ios::binary | std::ios::ate);
    size_t fsize = reader.tellg();
    reader.seekg(0, std::ios::beg);

    uint32_t ndims_u32;
    reader.read((char *)&ndims_u32, sizeof(uint32_t));
    reader.seekg(0, std::ios::beg);
    size_t ndims = (size_t)ndims_u32;
    // 先读取第一个向量的维数，再重置reader

    size_t npts = fsize / ((ndims * datasize) + sizeof(uint32_t));
    std::cout << fsize << " " << ndims << " " << datasize << std::endl;
    std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

    if (is_generate != 0) {
        generate_random_data_based_on_origin(argv[2], argv[3], generate_npts, 1, jump_npts);
        return 0;
    }

    target_npts = std::min(npts, target_npts);

    if (target_npts + jump_npts > npts) {
        std::cout << "Error: too much target and jump npts" << std::endl;
        exit(-1);
    }

    npts = std::min(npts, target_npts);


    size_t blk_size = 131072;   // todo: blk
    size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
    std::cout << "# blks: " << nblks << std::endl;
    std::ofstream writer(argv[3], std::ios::binary);
    int32_t npts_s32 = (int32_t)npts;
    int32_t ndims_s32 = (int32_t)ndims;


    writer.write((char *)&npts_s32, sizeof(int32_t));
    writer.write((char *)&ndims_s32, sizeof(int32_t));
    // 先写这两个数据

    size_t chunknpts = std::min(npts + jump_npts, blk_size);
    uint8_t *read_buf = new uint8_t[chunknpts * ((ndims * datasize) + sizeof(uint32_t))];
    uint8_t *write_buf = new uint8_t[chunknpts * ndims * datasize];

    for (size_t i = 0; i < nblks; i++)
    {
        size_t cblk_size = std::min(npts - i * blk_size, blk_size);
        if (datasize == sizeof(float))
        {
            block_convert_float(reader, writer, (float *)read_buf, (float *)write_buf, cblk_size, ndims, jump_npts);
        }
        else
        {
            block_convert_byte(reader, writer, read_buf, write_buf, cblk_size, ndims);
        }
        std::cout << "Block #" << i << " written" << std::endl;
    }

    delete[] read_buf;
    delete[] write_buf;

    reader.close();
    writer.close();
}

// float data/sift/sift_query.fvecs data/sift/sift_query.fbin 5000 0
// float data/sift/sift_query.fvecs data/sift/sift_train.fbin 5000 5000
// float data/sift/sift_base.fvecs data/sift/sift_learn.fbin 100000


// todo gist_base to query
// float data/gist/gist_base.fvecs data/gist/gist_query.fbin 5000 5000
// float data/gist/gist_base.fvecs data/gist/gist_learn.fbin 100000


// float data/test/test_query.fvecs data/test/test_train.fbin 5000 0
// float data/test/test_query.fvecs data/test/test_query.fbin 5000 5000
// float data/test/test_learn.fvecs data/test/test_learn.fbin 100000

// float data/test/test_query.fvecs data/test/test_train.fbin 5000 0
// float data/test/test_query.fvecs data/test/test_query.fbin 5000 5000