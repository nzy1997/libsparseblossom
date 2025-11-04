// CxxWrap 绑定模板 - 用于 SparseBlossom Julia 封装
// 基于 PyMatching 的 pybind11 绑定改编

#include "jlcxx/jlcxx.hpp"
// 注意：不需要 jlcxx/stl.hpp，std::vector 等会自动处理

// PyMatching 核心头文件
#include "sparse_blossom/driver/user_graph.h"
#include "sparse_blossom/driver/mwpm_decoding.h"
#include "sparse_blossom/matcher/mwpm.h"
#include "stim.h"

#include <vector>
#include <string>
#include <tuple>

// ============================================================================
// 辅助函数
// ============================================================================

namespace sparseblossom_jl {

// 合并策略枚举映射
pm::MERGE_STRATEGY merge_strategy_from_string(const std::string& strategy) {
    if (strategy == "disallow") return pm::DISALLOW;
    if (strategy == "independent") return pm::INDEPENDENT;
    if (strategy == "smallest-weight") return pm::SMALLEST_WEIGHT;
    if (strategy == "keep-original") return pm::KEEP_ORIGINAL;
    if (strategy == "replace") return pm::REPLACE;
    throw std::invalid_argument("Unknown merge strategy: " + strategy);
}

} // namespace sparseblossom_jl

// ============================================================================
// CxxWrap 模块定义
// ============================================================================

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    using namespace jlcxx;
    
    // ------------------------------------------------------------------------
    // UserGraph 类绑定
    // ------------------------------------------------------------------------
    
    auto user_graph_type = mod.add_type<pm::UserGraph>("UserGraph")
        .constructor<>()
        .constructor<size_t>()
        .constructor<size_t, size_t>();
    
    // 添加边的方法
    user_graph_type.method("add_edge", 
        [](pm::UserGraph& self, 
           size_t node1, size_t node2,
           const std::vector<size_t>& observables,
           double weight, 
           double error_probability,
           const std::string& merge_strategy) {
            
            if (std::abs(weight) > pm::MAX_USER_EDGE_WEIGHT) {
                throw std::invalid_argument(
                    "Weight exceeds maximum edge weight: " + 
                    std::to_string(pm::MAX_USER_EDGE_WEIGHT));
            }
            
            self.add_or_merge_edge(
                node1, node2, observables, 
                weight, error_probability,
                sparseblossom_jl::merge_strategy_from_string(merge_strategy));
        });
    
    // 添加边界边
    user_graph_type.method("add_boundary_edge",
        [](pm::UserGraph& self,
           size_t node,
           const std::vector<size_t>& observables,
           double weight,
           double error_probability,
           const std::string& merge_strategy) {
            
            if (std::abs(weight) > pm::MAX_USER_EDGE_WEIGHT) {
                throw std::invalid_argument(
                    "Weight exceeds maximum edge weight: " + 
                    std::to_string(pm::MAX_USER_EDGE_WEIGHT));
            }
            
            self.add_or_merge_boundary_edge(
                node, observables, 
                weight, error_probability,
                sparseblossom_jl::merge_strategy_from_string(merge_strategy));
        });
    
    // 图属性查询方法
    user_graph_type.method("get_num_nodes", &pm::UserGraph::get_num_nodes);
    user_graph_type.method("get_num_edges", &pm::UserGraph::get_num_edges);
    user_graph_type.method("get_num_observables", &pm::UserGraph::get_num_observables);
    user_graph_type.method("get_num_detectors", &pm::UserGraph::get_num_detectors);
    
    // 边查询方法
    user_graph_type.method("has_edge", &pm::UserGraph::has_edge);
    user_graph_type.method("has_boundary_edge", &pm::UserGraph::has_boundary_edge);
    
    // ------------------------------------------------------------------------
    // 解码方法
    // ------------------------------------------------------------------------
    
    // 单次解码
    user_graph_type.method("decode",
        [](pm::UserGraph& self,
           const std::vector<uint64_t>& detection_events,
           bool enable_correlations) {
            
            auto& mwpm = enable_correlations ? 
                         self.get_mwpm_with_search_graph() : 
                         self.get_mwpm();
            
            std::vector<uint8_t> obs_crossed(self.get_num_observables(), 0);
            pm::total_weight_int weight = 0;
            
            pm::decode_detection_events(
                mwpm, detection_events, 
                obs_crossed.data(), weight, 
                enable_correlations);
            
            double rescaled_weight = (double)weight / 
                                     mwpm.flooder.graph.normalising_constant;
            
            return std::make_tuple(obs_crossed, rescaled_weight);
        });
    
    // 解码到边数组
    user_graph_type.method("decode_to_edges",
        [](pm::UserGraph& self,
           const std::vector<uint64_t>& detection_events,
           bool enable_correlations) {
            
            auto& mwpm = self.get_mwpm_with_search_graph();
            std::vector<int64_t> edges;
            edges.reserve(detection_events.size() / 2);
            
            if (enable_correlations) {
                pm::decode_detection_events_to_edges_with_edge_correlations(
                    mwpm, detection_events, edges);
            } else {
                pm::decode_detection_events_to_edges(
                    mwpm, detection_events, edges);
            }
            
            return edges;
        });
    
    // ------------------------------------------------------------------------
    // 从检测器错误模型创建图
    // ------------------------------------------------------------------------
    
    mod.method("detector_error_model_to_matching_graph",
        [](const std::string& dem_string, bool enable_correlations) {
            auto dem = stim::DetectorErrorModel(dem_string.c_str());
            return pm::detector_error_model_to_user_graph(
                dem, enable_correlations, pm::NUM_DISTINCT_WEIGHTS);
        });
    
    // 从文件加载 DEM
    mod.method("detector_error_model_file_to_matching_graph",
        [](const std::string& dem_path, bool enable_correlations) {
            FILE* file = fopen(dem_path.c_str(), "r");
            if (file == nullptr) {
                throw std::runtime_error("Failed to open file: " + dem_path);
            }
            auto dem = stim::DetectorErrorModel::from_file(file);
            fclose(file);
            return pm::detector_error_model_to_user_graph(
                dem, enable_correlations, pm::NUM_DISTINCT_WEIGHTS);
        });
    
    // 从 Stim 电路文件加载
    mod.method("stim_circuit_file_to_matching_graph",
        [](const std::string& circuit_path, bool enable_correlations) {
            FILE* file = fopen(circuit_path.c_str(), "r");
            if (file == nullptr) {
                throw std::runtime_error("Failed to open file: " + circuit_path);
            }
            auto circuit = stim::Circuit::from_file(file);
            fclose(file);
            
            auto dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(
                circuit, true, true, false, 0, false, false);
            
            return pm::detector_error_model_to_user_graph(
                dem, enable_correlations, pm::NUM_DISTINCT_WEIGHTS);
        });
    
    // ------------------------------------------------------------------------
    // 辅助工具函数
    // ------------------------------------------------------------------------
    
    mod.method("merge_weights", &pm::merge_weights);
    
    // 导出常量
    mod.set_const("MAX_USER_EDGE_WEIGHT", pm::MAX_USER_EDGE_WEIGHT);
    mod.set_const("NUM_DISTINCT_WEIGHTS", pm::NUM_DISTINCT_WEIGHTS);
}

// ============================================================================
// 说明
// ============================================================================

/*
编译命令示例:

g++ -shared -fPIC -O3 -std=c++20 \
    -I/path/to/JlCxx/include \
    -I./src \
    cxxwrap_bindings.cc \
    -o libsparseblossom.so \
    -L/path/to/JlCxx/lib -lcxxwrap_julia \
    -lstim -lpthread

Julia 使用示例:

using CxxWrap
@wrapmodule("libsparseblossom")

# 创建图
graph = UserGraph(10, 2)

# 添加边
add_edge(graph, 0, 1, [0], 1.5, 0.1, "disallow")

# 解码
obs, weight = decode(graph, [0, 1, 2], false)

关键差异 (vs pybind11):
1. 使用 jlcxx::Module 代替 py::module
2. 返回 std::tuple 代替 py::tuple
3. 数组使用 std::vector 自动转换
4. 异常处理自动映射到 Julia
*/

