# CMakeLists.txt for libsparseblossom (Julia wrapper)
# 用于编译 SparseBlossom C++ 核心 + CxxWrap 绑定

cmake_minimum_required(VERSION 3.15)
project(libsparseblossom VERSION 2.3.0 LANGUAGES CXX)

# ============================================================================
# 编译选项
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 优化标志
if(NOT MSVC)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES x86_64)
        set(ARCH_OPT "-O3" "-mno-avx2")
    else()
        set(ARCH_OPT "-O3")
    endif()
else()
    set(ARCH_OPT "-O2")
endif()

# ============================================================================
# 查找依赖
# ============================================================================

# 查找 JlCxx (CxxWrap.jl 的 C++ 接口)
# 在 BinaryBuilder 环境中，FindJulia.cmake 会失败，所以我们手动创建目标
if(CMAKE_CROSSCOMPILING)
    message(STATUS "Cross-compiling detected, manually setting up JlCxx")
    # 查找 cxxwrap_julia 库
    find_library(CXXWRAP_JULIA_LIB cxxwrap_julia REQUIRED)
    find_path(CXXWRAP_JULIA_INCLUDE jlcxx/jlcxx.hpp REQUIRED)
    
    # 创建导入目标
    add_library(JlCxx::cxxwrap_julia SHARED IMPORTED)
    set_target_properties(JlCxx::cxxwrap_julia PROPERTIES
        IMPORTED_LOCATION "${CXXWRAP_JULIA_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${CXXWRAP_JULIA_INCLUDE}"
    )
    message(STATUS "JlCxx library: ${CXXWRAP_JULIA_LIB}")
    message(STATUS "JlCxx include: ${CXXWRAP_JULIA_INCLUDE}")
else()
    # 本地编译时使用正常的 find_package
    find_package(JlCxx REQUIRED)
    message(STATUS "Found JlCxx: ${JlCxx_DIR}")
endif()

# ============================================================================
# 获取 Stim 依赖 (通过 FetchContent)
# ============================================================================

include(FetchContent)

FetchContent_Declare(
    stim
    GIT_REPOSITORY https://github.com/quantumlib/Stim.git
    GIT_TAG 1320ad7eac7de34d2e9c70daa44fbc6d84174450
)

FetchContent_MakeAvailable(stim)

if(NOT MSVC)
    target_compile_options(libstim PRIVATE -fno-strict-aliasing -fPIC ${ARCH_OPT})
else()
    target_compile_options(libstim PRIVATE -fPIC ${ARCH_OPT})
endif()

# ============================================================================
# 源文件列表
# ============================================================================

# 包含目录
include_directories(src)
# 为了兼容 PyMatching 源码的 include 路径，添加映射
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
# 让 pymatching/xxx 路径能够找到 src/xxx
set(CMAKE_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Sparse Blossom 核心源文件
set(SPARSE_BLOSSOM_SOURCES
    # Driver
    src/sparse_blossom/driver/namespaced_main.cc
    src/sparse_blossom/driver/mwpm_decoding.cc
    src/sparse_blossom/driver/user_graph.cc
    
    # Flooder
    src/sparse_blossom/flooder/graph.cc
    src/sparse_blossom/flooder/detector_node.cc
    src/sparse_blossom/flooder/graph_fill_region.cc
    src/sparse_blossom/flooder/match.cc
    src/sparse_blossom/flooder/graph_flooder.cc
    
    # Matcher
    src/sparse_blossom/matcher/alternating_tree.cc
    src/sparse_blossom/matcher/mwpm.cc
    
    # Flooder-Matcher Interop
    src/sparse_blossom/flooder_matcher_interop/compressed_edge.cc
    src/sparse_blossom/flooder_matcher_interop/region_edge.cc
    src/sparse_blossom/flooder_matcher_interop/mwpm_event.cc
    
    # Tracker
    src/sparse_blossom/tracker/flood_check_event.cc
    
    # Diagram (可选，用于可视化)
    src/sparse_blossom/diagram/animation_main.cc
    src/sparse_blossom/diagram/mwpm_diagram.cc
    
    # Search
    src/sparse_blossom/search/search_graph.cc
    src/sparse_blossom/search/search_detector_node.cc
    src/sparse_blossom/search/search_flooder.cc
    
    # Random
    src/rand/rand_gen.cc
)

# ============================================================================
# 构建共享库
# ============================================================================

add_library(sparseblossom SHARED
    ${SPARSE_BLOSSOM_SOURCES}
    src/cxxwrap_bindings.cc  # CxxWrap 绑定文件
)

# 链接库
target_link_libraries(sparseblossom 
    PUBLIC 
        JlCxx::cxxwrap_julia
        libstim
)

if(NOT MSVC)
    target_link_libraries(sparseblossom PRIVATE pthread)
endif()

# 编译选项
target_compile_options(sparseblossom PRIVATE ${ARCH_OPT})

# 包含目录
target_include_directories(sparseblossom 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)

# ============================================================================
# 安装规则
# ============================================================================

install(TARGETS sparseblossom
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# 安装头文件
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/"
    DESTINATION include
    FILES_MATCHING 
    PATTERN "*.h"
    PATTERN "*.inl"
    PATTERN "*test.cc" EXCLUDE
    PATTERN "*perf.cc" EXCLUDE
    PATTERN "*pybind*" EXCLUDE
)

# 导出 CMake 配置（可选，用于其他项目）
# 注释掉，因为 libstim 不在导出集中
# install(EXPORT sparseblossomTargets
#     FILE sparseblossomTargets.cmake
#     NAMESPACE sparseblossom::
#     DESTINATION lib/cmake/sparseblossom
# )

# ============================================================================
# 测试（可选）
# ============================================================================

option(BUILD_TESTS "Build unit tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
    
    # 测试文件
    set(TEST_FILES
        src/sparse_blossom/driver/user_graph.test.cc
        src/sparse_blossom/flooder/graph.test.cc
        src/sparse_blossom/matcher/mwpm.test.cc
        # 添加更多测试...
    )
    
    add_executable(sparseblossom_tests 
        ${SPARSE_BLOSSOM_SOURCES}
        ${TEST_FILES}
    )
    
    target_link_libraries(sparseblossom_tests 
        GTest::gtest_main 
        libstim
    )
    
    include(GoogleTest)
    gtest_discover_tests(sparseblossom_tests)
endif()

# ============================================================================
# 信息输出
# ============================================================================

message(STATUS "==============================================")
message(STATUS "libsparseblossom configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  JlCxx found: YES")
message(STATUS "==============================================")

