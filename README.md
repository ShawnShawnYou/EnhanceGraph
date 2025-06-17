# EnhanceGraph: An Enhanced Graph-Based Index for High-Dimensional Approximate Nearest Neighbor Search
## Introduction 
- This repo is the implementation of "EnhanceGraph: An Enhanced Graph-Based Index for High-Dimensional Approximate Nearest Neighbor Search".
- We add features based on the [DISKANN](https://github.com/microsoft/DiskANN). Thus, the basic usage of this repo is consistent with DISKANN.
- The main function of EnhanceGraph is in /apps/experiment_memory_index.cpp
- To use it, there are several parameters.
    - dataset
    - train mode (input 1 for train -SC, input 2 for -S, input 3 for -C)
    - eval_mode (input 1 for -SC, input 2 for -S, input 3 for -C)
    - is_validate (validate on the training set)
    - algo_name (HNSW, VAMANA...)
    - dist_fn (mips, l2, cosine)
    - K (recall@k, typically 10)
    - train_L (L_2, typically same with the search parameter L, e.g., 100)
    - result_path (output result in this path)
    - build_L (L_1, typically 100)
    - build_R (maximum degree)
    - build_A (relaxation factor, typically 1.2)
    - delta_str (generate parameter, typically 0.51)
    - train_R (number of generated search points, typically 5)

## Running Instructions (EXAMPLE IN VSAG)
For a simple example, refer to examples/cpp/example_conjugate_graph.cpp in VSAG to see the recall rate enhancement directly.

If you want more details about our experiment, follow these instructions. (WARNING: The instructions are only examples. To run them, you need to configure many parameters and paths in the code. As it may take a lot of time to run and debug, we recommend running the example code in VSAG. :) )

## Running Instructions (FULL PIPLINE IMPLEMENTED IN DISKANN)
#### 1. ENVIRONMENT, PAPARE DATA AND PRE-PROCESS
- Environment: Consistent with DISKANN. You can follow their instructions to build the environment.
- Dataset: can be downloaded in [ANN-BENCHMARKS](https://github.com/erikbern/ann-benchmarks).
- Transform data from *fvecs* to *fbin*:
```
./apps/utils/fvecs_to_bin float gist-960-euclidean_learn.fvecs gist-960-euclidean_learn.fbin 10000000 0 0
```
- Compute ground truth
```
./apps/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file gist-960-euclidean_learn.fbin --query_file  gist-960-euclidean_query.fbin --gt_file gist-960-euclidean_query_learn_gt100 --K 100
```

#### 2. BUILD INDEX 
    ./apps/build_memory_index --data_type float --dist_fn l2  --data_path gist-960-euclidean_learn.fbin --index_path_prefix VAMANA_gist-960-euclidean_learn_R12_L100_A1.2 --result_path "" -R 12 -L 100 --alpha 1.2 --algorithm VAMANA --dataset gist-960-euclidean

#### 3. BUILD CONJUGATE GRAPH BY CONSTRUCTION LOG
    ./apps/experiment_memory_index gist-960-euclidean 3 0 0 VAMANA l2 10 100 "" 100 12 1.2 0.51 5

#### 4. TRAIN BY HISTORICAL AND GENERATED SEARCH LOG
    ./apps/experiment_memory_index gist-960-euclidean 1 0 0 VAMANA l2 10 100 "" 100 12 1.2 0.51 5

#### 5. QUERY WITHOUT CONJUGATE GRAPH
    ./apps/experiment_memory_index gist-960-euclidean 0 0 0 VAMANA l2 10 100 "" 100 12 1.2 0.51 5

#### 5. QUERY WITH CONJUGATE GRAPH
    ./apps/experiment_memory_index gist-960-euclidean 0 1 0 VAMANA l2 10 100 "" 100 12 1.2 0.51 5
