#pragma once
#include "distance.h"
#include "parameters.h"
#include "utils.h"
#include "types.h"
#include "index_config.h"
#include "index_build_params.h"
#include <any>

namespace diskann
{
struct consolidation_report
{
    enum status_code
    {
        SUCCESS = 0,
        FAIL = 1,
        LOCK_FAIL = 2,
        INCONSISTENT_COUNT_ERROR = 3
    };
    status_code _status;
    size_t _active_points, _max_points, _empty_slots, _slots_released, _delete_set_size, _num_calls_to_process_delete;
    double _time;

    consolidation_report(status_code status, size_t active_points, size_t max_points, size_t empty_slots,
                         size_t slots_released, size_t delete_set_size, size_t num_calls_to_process_delete,
                         double time_secs)
        : _status(status), _active_points(active_points), _max_points(max_points), _empty_slots(empty_slots),
          _slots_released(slots_released), _delete_set_size(delete_set_size),
          _num_calls_to_process_delete(num_calls_to_process_delete), _time(time_secs)
    {
    }
};

/* A templated independent class for intercation with Index. Uses Type Erasure to add virtual implemetation of methods
that can take any type(using std::any) and Provides a clean API that can be inherited by different type of Index.
*/
class AbstractIndex
{
  public:
    // setting
    bool use_bfs = false;
    bool use_knn_graph = true;
    bool use_many_knn_graph = false;
    bool use_extra_search = false;
    bool use_cached_top1 = true;
    bool use_aknng_enhancement = false;

    enum prune_strategy {
        VAMANA = 0,
        NSG = 1,
        TAUMNG = 2,
    };

    prune_strategy strategy = NSG;
    float tau = 0.04; // sift:10  gist:0.04



    // build setting
    int AKNNG_R = 32;   // AKNNG degree limit R
    int AKNNG_L = 100;   // AKNNG build search parameter L
    bool is_random_select_aknng_neighbor = false;

    // count result
    float inter_knng_final_count = 0;
    float inter_aknng_final_count = 0;
    float inter_knng2_final_count = 0;
    float add_success = 0;
    float valid_insert = 0;

    // KNNG
//    uint32_t *base_gt_ids = nullptr;
//    float *base_gt_dists = nullptr;
//    size_t base_gt_num = 0, base_gt_dim = 0;
//    size_t gt_dim = -1;
//    uint32_t *gt_ids = nullptr;
//
//
//    void get_base_gt_info(diskann::location_t base_id, std::vector<diskann::location_t>& gt_id_vec, std::vector<float>& gt_dist_vec) {
//        uint32_t *gt_id_vec_start = base_gt_ids + (uint32_t)base_gt_dim * base_id;
//        float* gt_dist_vec_start = base_gt_dists + (uint32_t)base_gt_dim * base_id;
//
//        gt_id_vec.assign(gt_id_vec_start, gt_id_vec_start + (uint32_t)base_gt_dim);
//        gt_dist_vec.assign(gt_dist_vec_start, gt_dist_vec_start + (uint32_t)base_gt_dim);
//    }

    virtual void get_data(float* data, diskann::location_t id) = 0;

    virtual const std::vector<diskann::location_t> &get_neighbors_dual(location_t id) = 0;
    virtual const std::vector<diskann::location_t> &get_neighbors(location_t id) = 0;

    virtual void add_neighbor(diskann::location_t out_id, diskann::location_t in_id) = 0;
    virtual void add_neighbor_dual(location_t out_id, location_t in_id) = 0;
    virtual float get_distance(location_t loc1, location_t loc2) = 0;
    virtual float get_distance(float* query, location_t loc2) = 0;



    AbstractIndex() = default;
    virtual ~AbstractIndex() = default;

    virtual void build(const std::string &data_file, const size_t num_points_to_load,
                       IndexFilterParams &build_params) = 0;

    template <typename data_type, typename tag_type>
    void build(const data_type *data, const size_t num_points_to_load, const std::vector<tag_type> &tags);

    virtual void save(const char *filename, bool compact_before_save = false) = 0;

#ifdef EXEC_ENV_OLS
    virtual void load(AlignedFileReader &reader, uint32_t num_threads, uint32_t search_l) = 0;
#else
    virtual void load(const char *index_file, uint32_t num_threads, uint32_t search_l) = 0;
#endif

    // For FastL2 search on optimized layout
    template <typename data_type>
    void search_with_optimized_layout(const data_type *query, size_t K, size_t L, uint32_t *indices);

    // Initialize space for res_vectors before calling.
    template <typename data_type, typename tag_type>
    size_t search_with_tags(const data_type *query, const uint64_t K, const uint32_t L, tag_type *tags,
                            float *distances, std::vector<data_type *> &res_vectors);

    // Added search overload that takes L as parameter, so that we
    // can customize L on a per-query basis without tampering with "Parameters"
    // IDtype is either uint32_t or uint64_t
    template <typename data_type, typename IDType>
    std::pair<uint32_t, uint32_t> search(const data_type *query, const size_t K, const uint32_t L, IDType *indices,
                                         float *distances = nullptr);


    template <typename data_type, typename IDType>
    std::pair<uint32_t, uint32_t> search_ret_route(int64_t i, const data_type *query, const size_t K, const uint32_t L,
                                                   IDType *indices, std::vector<IDType>& route, float *distances = nullptr);


    // Filter support search
    // IndexType is either uint32_t or uint64_t
    template <typename IndexType>
    std::pair<uint32_t, uint32_t> search_with_filters(const DataType &query, const std::string &raw_label,
                                                      const size_t K, const uint32_t L, IndexType *indices,
                                                      float *distances);

    template <typename data_type, typename tag_type> int insert_point(const data_type *point, const tag_type tag);

    template <typename tag_type> int lazy_delete(const tag_type &tag);

    template <typename tag_type>
    void lazy_delete(const std::vector<tag_type> &tags, std::vector<tag_type> &failed_tags);

    template <typename tag_type> void get_active_tags(tsl::robin_set<tag_type> &active_tags);

    template <typename data_type> void set_start_points_at_random(data_type radius, uint32_t random_seed = 0);

    virtual consolidation_report consolidate_deletes(const IndexWriteParameters &parameters) = 0;

    virtual void optimize_index_layout() = 0;

    // memory should be allocated for vec before calling this function
    template <typename tag_type, typename data_type> int get_vector_by_tag(tag_type &tag, data_type *vec);

  private:
    virtual void _build(const DataType &data, const size_t num_points_to_load, TagVector &tags) = 0;
    virtual std::pair<uint32_t, uint32_t> _search(const DataType &query, const size_t K, const uint32_t L,
                                                  std::any &indices, float *distances = nullptr) = 0;

    virtual std::pair<uint32_t, uint32_t> _search_ret_route(int64_t i, const DataType &query, const size_t K, const uint32_t L,
                                                            std::any &indices, std::any &route, float *distances = nullptr) = 0;


    virtual std::pair<uint32_t, uint32_t> _search_with_filters(const DataType &query, const std::string &filter_label,
                                                               const size_t K, const uint32_t L, std::any &indices,
                                                               float *distances) = 0;
    virtual int _insert_point(const DataType &data_point, const TagType tag) = 0;
    virtual int _lazy_delete(const TagType &tag) = 0;
    virtual void _lazy_delete(TagVector &tags, TagVector &failed_tags) = 0;
    virtual void _get_active_tags(TagRobinSet &active_tags) = 0;
    virtual void _set_start_points_at_random(DataType radius, uint32_t random_seed = 0) = 0;
    virtual int _get_vector_by_tag(TagType &tag, DataType &vec) = 0;
    virtual size_t _search_with_tags(const DataType &query, const uint64_t K, const uint32_t L, const TagType &tags,
                                     float *distances, DataVector &res_vectors) = 0;
    virtual void _search_with_optimized_layout(const DataType &query, size_t K, size_t L, uint32_t *indices) = 0;
};
} // namespace diskann
