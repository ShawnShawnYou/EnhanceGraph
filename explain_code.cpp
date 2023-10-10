//
// Created by Shawn on 2023/9/5.

// 1. 图的结构怎么存放的？ -- fixed
std::vector<std::vector<uint32_t>> _graph;

// 2. link函数怎么连接的？ -- fixed

// 3. scratch是什么东西？



// leanring
node_list.clear();
node_list.shrink_to_fit();




// 关键变量：
location_t i;   // 是指在图中id为i

node; // 当前处理的节点id
gt_file; // ground truth file

// 关键函数：

link = 1 + 2 + 3;



1. search_for_point_and_prune // 找节点 + prune
{

    1.11 iterate_to_fixed_point;

    1.12 prune_neighbors {
        1.21 occlude_list;
    }

};

2. set // 增加边
{
    2.1 LockGuard guard(_locks[node]);
    2.2 _graph_store->set_neighbours(node, pruned_list);
};

3. inter_insert; // 维护被增加边的度数平衡


merge_shards


// search

cached_beam_search;

medoid???

