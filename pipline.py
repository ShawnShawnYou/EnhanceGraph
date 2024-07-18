
import struct
import h5py
import random
import os

def load_h5py(filename, query_size, base_size):
    query = []
    groundtruth_id = []
    base = []

    file = h5py.File(filename, "r")
    for key in file.keys():
        if key == "query":
            query = file[key]
        elif key == "groundtruth":
            groundtruth_id = file[key]
        elif key == "base":
            base = file[key]

    gt_set = set()
    for i in range(0, len(query)):
        gt_id = int(groundtruth_id[i])
        gt_set.add(gt_id)

    assert base_size >= len(gt_set), f"asked base size {base_size} < ground truth size {len(gt_set)}"
    assert base_size <= len(base), f"asked base size {base_size} > base size {len(base)}"

    used_base_id = dict()                       # key: before gt_id     value: after insert map_id
    unused_base_id = set(range(len(base)))      # 没有使用过的base向量的id
    unused_map_id = set(range(base_size))       # 没有插入过的位置

    generated_base = [[] for i in range(base_size)]
    generated_groundtruth_id = [-1 for i in range(query_size)]

    for i in range(0, len(query)):
        gt_id = int(groundtruth_id[i])
        if gt_id not in used_base_id:
            # gt_id映射到random_map_id这个位置
            map_id = random.sample(unused_map_id, 1)[0]
            unused_map_id.remove(map_id)

            # 插入
            generated_base[map_id] = base[gt_id]
            generated_groundtruth_id[i] = map_id

            # 记录gt_id这base向量被用过了
            used_base_id[gt_id] = map_id
            unused_base_id.remove(gt_id)
        else:
            generated_groundtruth_id[i] = used_base_id[gt_id]


    for i in range(len(generated_base)):
        if len(generated_base[i]) == 0:
            # 选择一个没有用过的base向量
            random_base_id = random.sample(unused_base_id, 1)[0]
            unused_base_id.remove(random_base_id)

            # random_base_id这个base向量映射到了i这个位置
            used_base_id[random_base_id] = i
            generated_base[i] = base[random_base_id]

    return query, generated_groundtruth_id, generated_base, len(base), len(query)


def save_arrays_to_binary(filename, vector_list):
    # 打开文件
    with open(filename, 'wb') as f:
        # 遍历每个数组
        for vector in vector_list:
            # 获取数组的维度
            d = vector.shape[0]
            # 写入数组的维度
            f.write(struct.pack('I', d))

            # 遍历数组中的每个向量
            for element in vector:
                # 将每个向量的元素转换为float，并写入文件
                f.write(struct.pack('f', element))



def save_fvec(h5py_file_path, base_path, query_path, base_size=100000, query_size=10000):
    query, groundtruth_id, base, len_base, len_query = load_h5py(h5py_file_path, query_size, base_size)
    save_arrays_to_binary(base_path, base)
    save_arrays_to_binary(query_path, query)

    return


def pipeline(algo_name="vamana", dataset="gist1m", R=32, L=50, A=1.2):
    h5py_file_path = "/app/DiskANN/build/data/test/test.h5"
    base_size = 100000
    query_size = 10000

    work_dir = "/app/DiskANN/build/"

    if algo_name == "hnsw":
        app_path_prefix = "/root/algorithm/hnsw/cmake-build-debug-container_diskann/"
    else:
        app_path_prefix = "/app/DiskANN/cmake-build-debug-container_diskann/apps/"

    data_path_prefix = "/root/dataset/data/" + dataset + "/" + dataset
    index_path_prefix = "/root/index/" + algo_name  \
                        + "/index_" + daindex_path_prefixtaset + "_learn" \
                        + "_R" + str(R) + " +_L" + str(L) + "_A" + str(A)

    pg_path = index_path_prefix + ".pg"
    dg_path = index_path_prefix + ".dg"
    tg_path = index_path_prefix + "_train.dg"

    learn_fvecs = data_path_prefix + "_learn.fvecs"
    query_fvecs = data_path_prefix + "_query.fvecs"

    learn_fbin = data_path_prefix + "_learn.fbin"
    query_fbin = data_path_prefix + "_query.fbin"
    train_fbin = data_path_prefix + "_train.fbin"

    query_gt = data_path_prefix + "_query_learn_gt100"
    train_gt = data_path_prefix + "_train_learn_gt100"




    app_fvecs_to_bin = app_path_prefix + "utils/fvecs_to_bin"
    param_fvecs_to_bin_learn = "float" + " " + learn_fvecs + " " + learn_fbin + " " + str(base_size)
    param_fvecs_to_bin_query = "float" + " " + query_fvecs + " " + query_fbin + " " + str(int(query_size / 2)) + " " + str(0)
    param_fvecs_to_bin_train = "float" + " " + query_fvecs + " " + train_fbin + " " + str(int(query_size / 2)) + " " + str(int(query_size / 2))

    # param_fvecs_to_bin_learn = "float" + " " + learn_fvecs + " " + learn_fbin + " " + str(base_size) + " " + str(10000)
    # param_fvecs_to_bin_query = "float" + " " + learn_fvecs + " " + query_fbin + " " + str(int(query_size / 2)) + " " + str(10000)
    # param_fvecs_to_bin_train = "float" + " " + learn_fvecs + " " + train_fbin + " " + str(int(query_size / 2)) + " " + str(10000 + int(query_size / 2))


    app_compute_gt = app_path_prefix + "utils/compute_groundtruth"
    param_compute_gt_query = " --data_type float " \
                             " --dist_fn l2 " \
                             " --base_file " + learn_fbin + \
                             " --query_file  " + query_fbin + \
                             " --gt_file " + query_gt + \
                             " --K 100 "

    param_compute_gt_train = " --data_type float " \
                             " --dist_fn l2 " \
                             " --base_file " + learn_fbin + \
                             " --query_file  " + train_fbin + \
                             " --gt_file " + train_gt + \
                             " --K 100 "

    app_build_index = app_path_prefix + "build_memory_index"
    param_build_index = " --data_type float" \
                        " --dist_fn l2" \
                        " --data_path " + learn_fbin + \
                        " --index_path_prefix " + index_path + \
                        " -R 32" \
                        " -L 50" \
                        " --alpha 1.2"

    app_train = app_path_prefix + "train_memory_index"
    param_train = dataset + " 1 0 0 "
    param_valid = dataset + " 0 0 1 "
    param_query = dataset + " 0 0 0 "
    param_eval  = dataset + " 0 1 0 "


    # save_fvec(h5py_file_path, learn_fvecs, query_fvecs)

    # 1. change work dir
    os.chdir(work_dir)
    print("Finish Phase 1")

    # 2. fvecs to fbin
    # os.system(app_fvecs_to_bin + " " + param_fvecs_to_bin_learn)
    # os.system(app_fvecs_to_bin + " " + param_fvecs_to_bin_query)
    # os.system(app_fvecs_to_bin + " " + param_fvecs_to_bin_train)
    print("Finish Phase 2")

    # 3. compute gt
    # os.system(app_compute_gt + " " + param_compute_gt_query)
    # print(app_compute_gt + " " + param_compute_gt_query)
    # os.system(app_compute_gt + " " + param_compute_gt_train)
    print("Finish Phase 3")

    # 4. build index
    os.system(app_build_index + " " + param_build_index)
    print(app_build_index + " " + param_build_index)
    print("Finish Phase 4")

    # 5. train and query
    # os.system(app_train + " " + param_train)
    # os.system(app_train + " " + param_valid)
    # os.system(app_train + " " + param_query)
    # os.system(app_train + " " + param_eval)
    print("Finish Phase 5")





if __name__ == '__main__':
    pipeline()
