#include "index_factory.h"

namespace diskann
{

IndexFactory::IndexFactory(const IndexConfig &config) : _config(std::make_unique<IndexConfig>(config))
{
    check_config();
}

std::unique_ptr<AbstractIndex> IndexFactory::create_instance()
{
    return create_instance(_config->data_type, _config->tag_type, _config->label_type);
}

void IndexFactory::check_config()
{
    if (_config->dynamic_index && !_config->enable_tags)
    {
        throw ANNException("ERROR: Dynamic Indexing must have tags enabled.", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (_config->pq_dist_build)
    {
        if (_config->dynamic_index)
            throw ANNException("ERROR: Dynamic Indexing not supported with PQ distance based "
                               "index construction",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        if (_config->metric == diskann::Metric::INNER_PRODUCT)
            throw ANNException("ERROR: Inner product metrics not yet supported "
                               "with PQ distance "
                               "base index",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (_config->data_type != "float" && _config->data_type != "uint8" && _config->data_type != "int8")
    {
        throw ANNException("ERROR: invalid data type : + " + _config->data_type +
                               " is not supported. please select from [float, int8, uint8]",
                           -1);
    }

    if (_config->tag_type != "int32" && _config->tag_type != "uint32" && _config->tag_type != "int64" &&
        _config->tag_type != "uint64")
    {
        throw ANNException("ERROR: invalid data type : + " + _config->tag_type +
                               " is not supported. please select from [int32, uint32, int64, uint64]",
                           -1);
    }
}

template <typename T>
std::unique_ptr<AbstractDataStore<T>> IndexFactory::construct_datastore(const DataStoreStrategy strategy,
                                                                        const size_t num_points, const size_t dimension,
                                                                        const Metric m)
{
    std::unique_ptr<Distance<T>> distance;
    switch (strategy)
    {
    case diskann::DataStoreStrategy::MEMORY:
        if (m == diskann::Metric::COSINE && std::is_same<T, float>::value)
        {
            distance.reset((Distance<T> *)new AVXNormalizedCosineDistanceFloat());
            return std::make_unique<diskann::InMemDataStore<T>>((location_t)num_points, dimension, std::move(distance));
        }
        else
        {
            distance.reset((Distance<T> *)get_distance_function<T>(m));
            return std::make_unique<diskann::InMemDataStore<T>>((location_t)num_points, dimension, std::move(distance));
        }
        break;
    default:
        break;
    }
    return nullptr;
}

std::unique_ptr<AbstractGraphStore> IndexFactory::construct_graphstore(const GraphStoreStrategy strategy,
                                                                       const size_t size,
                                                                       const size_t reserve_graph_degree)
{
    switch (strategy)
    {
    case GraphStoreStrategy::MEMORY:
        return std::make_unique<InMemGraphStore>(size, reserve_graph_degree);
    default:
        throw ANNException("Error : Current GraphStoreStratagy is not supported.", -1);
    }
}

template <typename data_type, typename tag_type, typename label_type>
std::unique_ptr<AbstractIndex> IndexFactory::create_instance()
{
    size_t num_points = _config->max_points + _config->num_frozen_pts;
    size_t dim = _config->dimension;
    size_t max_reserve_degree =
        (size_t)(defaults::GRAPH_SLACK_FACTOR * 1.05 *
                 (_config->index_write_params == nullptr ? 0 : _config->index_write_params->max_degree));
    auto data_store = construct_datastore<data_type>(_config->data_strategy, num_points, dim, _config->metric);
    auto graph_store =
        construct_graphstore(_config->graph_strategy, num_points + _config->num_frozen_pts, max_reserve_degree);
    auto dual_graph_store =
            construct_graphstore(_config->graph_strategy, num_points + _config->num_frozen_pts, max_reserve_degree);

    auto ret = std::make_unique<diskann::Index<data_type, tag_type, label_type>>(*_config, std::move(data_store),
                                                                             std::move(graph_store));
    ret->init_dual_graph_store(std::move(dual_graph_store));

    return ret;
}

std::unique_ptr<AbstractIndex> IndexFactory::create_instance(const std::string &data_type, const std::string &tag_type,
                                                             const std::string &label_type)
{
    if (data_type == std::string("float"))
    {
        return create_instance<float>(tag_type, label_type);
    }
    else if (data_type == std::string("uint8"))
    {
        return create_instance<uint8_t>(tag_type, label_type);
    }
    else if (data_type == std::string("int8"))
    {
        return create_instance<int8_t>(tag_type, label_type);
    }
    else
        throw ANNException("Error: unsupported data_type please choose from [float/int8/uint8]", -1);
}

template <typename data_type>
std::unique_ptr<AbstractIndex> IndexFactory::create_instance(const std::string &tag_type, const std::string &label_type)
{
    if (tag_type == std::string("int32"))
    {
        return create_instance<data_type, int32_t>(label_type);
    }
    else if (tag_type == std::string("uint32"))
    {
        return create_instance<data_type, uint32_t>(label_type);
    }
    else if (tag_type == std::string("int64"))
    {
        return create_instance<data_type, int64_t>(label_type);
    }
    else if (tag_type == std::string("uint64"))
    {
        return create_instance<data_type, uint64_t>(label_type);
    }
    else
        throw ANNException("Error: unsupported tag_type please choose from [int32/uint32/int64/uint64]", -1);
}

template <typename data_type, typename tag_type>
std::unique_ptr<AbstractIndex> IndexFactory::create_instance(const std::string &label_type)
{
    if (label_type == std::string("uint16") || label_type == std::string("ushort"))
    {
        return create_instance<data_type, tag_type, uint16_t>();
    }
    else if (label_type == std::string("uint32") || label_type == std::string("uint"))
    {
        return create_instance<data_type, tag_type, uint32_t>();
    }
    else
        throw ANNException("Error: unsupported label_type please choose from [uint/ushort]", -1);
}

template DISKANN_DLLEXPORT std::unique_ptr<AbstractDataStore<uint8_t>> IndexFactory::construct_datastore(
    DataStoreStrategy stratagy, size_t num_points, size_t dimension, Metric m);
template DISKANN_DLLEXPORT std::unique_ptr<AbstractDataStore<int8_t>> IndexFactory::construct_datastore(
    DataStoreStrategy stratagy, size_t num_points, size_t dimension, Metric m);
template DISKANN_DLLEXPORT std::unique_ptr<AbstractDataStore<float>> IndexFactory::construct_datastore(
    DataStoreStrategy stratagy, size_t num_points, size_t dimension, Metric m);

} // namespace diskann
