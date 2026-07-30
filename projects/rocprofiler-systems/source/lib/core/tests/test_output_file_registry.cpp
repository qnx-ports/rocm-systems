// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/output_file_registry.hpp"

#include <string_view>

namespace
{
using rocprofsys::output_file;
using rocprofsys::output_file_registry;
using rocprofsys::output_format;

void
expect_entry(const output_file& entry, std::string_view label, std::string_view path,
             std::string_view viewer)
{
    EXPECT_EQ(entry.label, label);
    EXPECT_EQ(entry.path, path);
    EXPECT_EQ(entry.viewer, viewer);
}
}  // namespace

TEST(output_file_registry, make_entry_perfetto)
{
    expect_entry(
        output_file_registry::make_entry("/tmp/trace.proto", output_format::perfetto),
        "Perfetto trace", "/tmp/trace.proto", "Open in https://ui.perfetto.dev");
}

TEST(output_file_registry, make_entry_rocpd)
{
    expect_entry(
        output_file_registry::make_entry("/tmp/profile.db", output_format::rocpd),
        "RocPD database", "/tmp/profile.db",
        "sqlite3, ROCm Optiq, or rocprofiler-sdk provided rocpd "
        "Python module for conversion to other formats");
}

TEST(output_file_registry, make_entry_json_without_component)
{
    expect_entry(output_file_registry::make_entry("/tmp/out.json", output_format::json),
                 "JSON output", "/tmp/out.json", "jq . /tmp/out.json");
}

TEST(output_file_registry, make_entry_json_with_component)
{
    expect_entry(
        output_file_registry::make_entry("/tmp/out.json", output_format::json, "mpi"),
        "JSON (mpi)", "/tmp/out.json", "jq . /tmp/out.json");
}

TEST(output_file_registry, make_entry_text_without_component)
{
    expect_entry(output_file_registry::make_entry("/tmp/out.txt", output_format::text),
                 "Text profile", "/tmp/out.txt", "cat /tmp/out.txt");
}

TEST(output_file_registry, make_entry_text_with_component)
{
    expect_entry(
        output_file_registry::make_entry("/tmp/out.txt", output_format::text, "gpu"),
        "Profile (gpu)", "/tmp/out.txt", "cat /tmp/out.txt");
}

TEST(output_file_registry, make_entry_causal_json)
{
    expect_entry(
        output_file_registry::make_entry("/tmp/causal.json", output_format::causal_json),
        "Causal profile (JSON)", "/tmp/causal.json", "jq . /tmp/causal.json");
}

TEST(output_file_registry, make_entry_causal_text)
{
    expect_entry(
        output_file_registry::make_entry("/tmp/causal.txt", output_format::causal_text),
        "Causal profile (text)", "/tmp/causal.txt", "cat /tmp/causal.txt");
}
