# README

- This repo is the implementation of "EnhanceGraph: A Continuously Enhanced Graph-Based Index for High-Dimensional Approximate Nearest Neighbor Search".
- We add features based on the DISKANN library. Thus, the basic usage of this repo is consistent with DISKANN.
- The main function of EnhanceGraph is in /apps/experiment_memory_index.cpp
- To use it, there are several parameters.
    - dataset
    - is_train (train mode)
    - eval_mode (input 1 for -SC, input 2 for -S, input 3 for -C)
    - is_validate (validate on the training set)
    - algo_name (HNSW, VAMANA...)
    - dist_fn (mips, l2, cosine)
    - K
    - train_L (L_2)
    - result_path (output result in this path)
    - build_L (L_1)
    - build_R (maximum degree)
    - build_A (relaxation factor, typically 1.2)
    - delta_str (generate parameter, typically 0.51)
    - train_R (number of generated search points)

