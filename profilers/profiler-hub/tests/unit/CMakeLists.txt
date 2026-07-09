# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(gtest)

set(UNIT_TEST_SOURCES
    writer_test.cpp
    reader_test.cpp
    insert_query_builders_test.cpp
    table_insert_query_test.cpp
    table_select_query_test.cpp
    database_test.cpp
    entity_utility_test.cpp
    json_serializer_test.cpp
    mock_sqlite3_test.cpp
    backend_over_mock_test.cpp
)

add_executable(profiler-hub_unit_tests ${UNIT_TEST_SOURCES})

target_link_libraries(
    profiler-hub_unit_tests
    PRIVATE
        profiler-hub
        profiler-hub_queries
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        profiler-hub-sqlite3
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

# v3 fixture (bundled).
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/rocpd.db
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd.db
    COPYONLY
)
set(ROCPD_DB_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd.db)

# Real v4.0 fixture (schema-verified genuine v4.0; committed as rocpd_v4.db).
# Covers cpu_thread/gpu_queue/dma interval tracks + region->kernel_dispatch/
# memory_copy flows. It has no rocpd_sample rows, so it does NOT exercise the
# scalar/counter path -- that gap is filled by the synthetic fixture below.
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/rocpd_v4.db
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v4.db
    COPYONLY
)
set(ROCPD_DB_V4_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v4.db)

# Synthetic v4.0 counter/scalar fixture, generated at build time from committed
# SQL (reproducible, reviewable in diff, no opaque binary). The canonical v4.0
# schema DDL and the hand-authored counter data live in fixtures/; the {{uuid}}/
# {{guid}} placeholders are substituted exactly as get_schema_query() does, then
# fed through the sqlite3 CLI. Regenerate manually with, e.g.:
#   uuid=00000000000000000000000000c00001
#   sed -e "s/{{uuid}}/_$uuid/g" -e "s/{{guid}}/$uuid/g" \
#       fixtures/rocpd_v4.0_tables.sql fixtures/rocpd_v4_counter_data.sql \
#     | sqlite3 rocpd_v4_counter.db
find_program(SQLITE3_EXECUTABLE sqlite3)
if(NOT SQLITE3_EXECUTABLE)
    message(
        FATAL_ERROR
        "sqlite3 CLI not found; it is required to build the synthetic v4.0 "
        "counter test fixture (tests/unit/fixtures/rocpd_v4_counter_data.sql)."
    )
endif()

set(_v4c_uuid "00000000000000000000000000c00001")
set(_v4c_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v4_counter_gen.sql)
file(READ ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql _v4c_schema)
file(
    READ ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4_counter_data.sql
    _v4c_data
)
string(REPLACE "{{uuid}}" "_${_v4c_uuid}" _v4c_schema "${_v4c_schema}")
string(REPLACE "{{guid}}" "${_v4c_uuid}" _v4c_schema "${_v4c_schema}")
string(REPLACE "{{uuid}}" "_${_v4c_uuid}" _v4c_data "${_v4c_data}")
string(REPLACE "{{guid}}" "${_v4c_uuid}" _v4c_data "${_v4c_data}")
file(WRITE ${_v4c_gen_sql} "${_v4c_schema}\n${_v4c_data}\n")

set(ROCPD_DB_V4_COUNTER_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v4_counter.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V4_COUNTER_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V4_COUNTER_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V4_COUNTER_PATH}
        ".read ${_v4c_gen_sql}"
    DEPENDS
        ${_v4c_gen_sql}
        ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4.0_tables.sql
        ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v4_counter_data.sql
    COMMENT "Generating synthetic v4.0 counter/scalar test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v4_counter_fixture
    DEPENDS ${ROCPD_DB_V4_COUNTER_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v4_counter_fixture)

# Synthetic v3 edge-matrix fixture, generated at build time from committed SQL.
# Unlike the v4.0 counter fixture (which had to bundle its own schema copy because
# no canonical v4.0 DDL lives in-tree), the v3 schema IS canonical at
# source/data_storage/schema/rocpd_tables.sql, so we reference it directly to
# avoid a drifting duplicate. The {{uuid}}/{{guid}} placeholders are substituted
# exactly as get_schema_query() does, then fed through the sqlite3 CLI. The db has
# no rocpd_timestamp table, so the reader selects the v3 backend. Regenerate with:
#   uuid=00000000000000000000000000e30003
#   sed -e "s/{{uuid}}/_$uuid/g" -e "s/{{guid}}/$uuid/g" \
#       ../../source/data_storage/schema/rocpd_tables.sql \
#       fixtures/rocpd_v3_edge_data.sql | sqlite3 rocpd_v3_edge.db
set(_v3e_uuid "00000000000000000000000000e30003")
set(_v3e_schema_src
    ${CMAKE_SOURCE_DIR}/source/data_storage/schema/rocpd_tables.sql
)
set(_v3e_data_src ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v3_edge_data.sql)
set(_v3e_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v3_edge_gen.sql)
file(READ ${_v3e_schema_src} _v3e_schema)
file(READ ${_v3e_data_src} _v3e_data)
# The canonical schema ends with an INSERT INTO rocpd_metadata that uses
# double-quoted string literals. The modern sqlite3 CLI rejects double-quoted
# strings in DML (only DDL keeps the legacy behavior), and the reader never reads
# metadata anyway (v3 is detected by the ABSENCE of rocpd_timestamp). Strip that
# single trailing DML statement so the CLI is fed pure DDL, matching how the v4
# fixture schema is pure DDL. [^;] spans newlines in CMake regex.
string(
    REGEX REPLACE
    "INSERT INTO[^;]*rocpd_metadata[^;]*;"
    ""
    _v3e_schema
    "${_v3e_schema}"
)
string(REPLACE "{{uuid}}" "_${_v3e_uuid}" _v3e_schema "${_v3e_schema}")
string(REPLACE "{{guid}}" "${_v3e_uuid}" _v3e_schema "${_v3e_schema}")
string(REPLACE "{{uuid}}" "_${_v3e_uuid}" _v3e_data "${_v3e_data}")
string(REPLACE "{{guid}}" "${_v3e_uuid}" _v3e_data "${_v3e_data}")
file(WRITE ${_v3e_gen_sql} "${_v3e_schema}\n${_v3e_data}\n")

set(ROCPD_DB_V3_EDGE_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v3_edge.db)
add_custom_command(
    OUTPUT ${ROCPD_DB_V3_EDGE_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V3_EDGE_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V3_EDGE_PATH} ".read ${_v3e_gen_sql}"
    DEPENDS ${_v3e_gen_sql} ${_v3e_schema_src} ${_v3e_data_src}
    COMMENT "Generating synthetic v3 edge-matrix test fixture"
    VERBATIM
)
add_custom_target(profiler-hub_v3_edge_fixture DEPENDS ${ROCPD_DB_V3_EDGE_PATH})
add_dependencies(profiler-hub_unit_tests profiler-hub_v3_edge_fixture)

# Synthetic v3 dma-by-destination-agent fixture, generated at build time from
# committed SQL against the canonical v3 schema (same mechanism as the edge
# fixture above). Encodes the crossed 2-agent x 2-stream x 12 = 48 memory_copy
# pattern that proves dma tracks partition by dst_agent_id, not stream_id.
set(_v3d_uuid "00000000000000000000000000d30003")
set(_v3d_schema_src
    ${CMAKE_SOURCE_DIR}/source/data_storage/schema/rocpd_tables.sql
)
set(_v3d_data_src ${CMAKE_CURRENT_LIST_DIR}/fixtures/rocpd_v3_dma_agent_data.sql)
set(_v3d_gen_sql ${CMAKE_CURRENT_BINARY_DIR}/rocpd_v3_dma_agent_gen.sql)
file(READ ${_v3d_schema_src} _v3d_schema)
file(READ ${_v3d_data_src} _v3d_data)
# Strip the trailing rocpd_metadata DML (double-quoted string literals the modern
# sqlite3 CLI rejects in DML); the reader never reads metadata. See the edge block.
string(
    REGEX REPLACE
    "INSERT INTO[^;]*rocpd_metadata[^;]*;"
    ""
    _v3d_schema
    "${_v3d_schema}"
)
string(REPLACE "{{uuid}}" "_${_v3d_uuid}" _v3d_schema "${_v3d_schema}")
string(REPLACE "{{guid}}" "${_v3d_uuid}" _v3d_schema "${_v3d_schema}")
string(REPLACE "{{uuid}}" "_${_v3d_uuid}" _v3d_data "${_v3d_data}")
string(REPLACE "{{guid}}" "${_v3d_uuid}" _v3d_data "${_v3d_data}")
file(WRITE ${_v3d_gen_sql} "${_v3d_schema}\n${_v3d_data}\n")

set(ROCPD_DB_V3_DMA_AGENT_PATH
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rocpd_v3_dma_agent.db
)
add_custom_command(
    OUTPUT ${ROCPD_DB_V3_DMA_AGENT_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${ROCPD_DB_V3_DMA_AGENT_PATH}
    COMMAND
        ${SQLITE3_EXECUTABLE} ${ROCPD_DB_V3_DMA_AGENT_PATH}
        ".read ${_v3d_gen_sql}"
    DEPENDS ${_v3d_gen_sql} ${_v3d_schema_src} ${_v3d_data_src}
    COMMENT "Generating synthetic v3 dma-by-destination-agent test fixture"
    VERBATIM
)
add_custom_target(
    profiler-hub_v3_dma_agent_fixture
    DEPENDS ${ROCPD_DB_V3_DMA_AGENT_PATH}
)
add_dependencies(profiler-hub_unit_tests profiler-hub_v3_dma_agent_fixture)

target_compile_definitions(
    profiler-hub_unit_tests
    PUBLIC
        ROCPD_DB_PATH="${ROCPD_DB_PATH}"
        ROCPD_DB_V4_PATH="${ROCPD_DB_V4_PATH}"
        ROCPD_DB_V4_COUNTER_PATH="${ROCPD_DB_V4_COUNTER_PATH}"
        ROCPD_DB_V3_EDGE_PATH="${ROCPD_DB_V3_EDGE_PATH}"
        ROCPD_DB_V3_DMA_AGENT_PATH="${ROCPD_DB_V3_DMA_AGENT_PATH}"
)

target_include_directories(
    profiler-hub_unit_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/source
        ${CMAKE_SOURCE_DIR}/source/common
        ${CMAKE_SOURCE_DIR}/source/data_storage
        ${CMAKE_SOURCE_DIR}/source/queries
)

# backend_over_mock_test.cpp includes sqlite_backend_impl.hpp directly to
# instantiate database_backend over the mock policy, so this target needs
# the same schema configuration as profiler-hub-objects (see top-level
# CMakeLists.txt) for the impl header to resolve the correct branch.
if(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD)
    target_link_libraries(
        profiler-hub_unit_tests
        PRIVATE rocprofiler-sdk-rocpd::rocprofiler-sdk-rocpd
    )

    target_compile_definitions(
        profiler-hub_unit_tests
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=1>
    )
else()
    target_include_directories(
        profiler-hub_unit_tests
        PRIVATE ${SQL_SCHEMA_BINARY_DIR}
    )

    target_compile_definitions(
        profiler-hub_unit_tests
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=0>
    )
endif()

target_compile_features(profiler-hub_unit_tests PRIVATE cxx_std_17)

if(PROFILER_HUB_ENABLE_COVERAGE)
    profiler_hub_add_coverage_flags(profiler-hub_unit_tests)
endif()

gtest_discover_tests(
    profiler-hub_unit_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES LABELS "unit"
)
