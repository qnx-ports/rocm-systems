// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/reader.hpp"
#include "profiler-hub/storage.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace
{

// --------------------------------------------------------------------------
// Track-scoped API test helpers (shared by v3 / v4 / v4-counter fixtures).
// --------------------------------------------------------------------------

// Mint an opaque event handle. Production consumers never construct these -- they
// receive them from the reader -- but the reader's own unit tests mint handles to
// exercise the typed detail accessors and prove handle disambiguation.
profiler_hub::reader_types::event_id_t
make_event_id(profiler_hub::reader_types::event_type_t type, size_t row_id)
{
    return profiler_hub::reader_types::detail::event_id_access::make(type, row_id);
}

// Peek the per-type row id an opaque handle routes to. Test-only: the public API
// treats event_id_t as opaque (equality / ordering / hashing only).
size_t
row_id_of(const profiler_hub::reader_types::event_id_t& id)
{
    return profiler_hub::reader_types::detail::event_id_access::row_id(id);
}

// Count how many of the four interval-detail accessors resolve a handle. A
// well-formed interval handle must resolve through exactly one; this replaces the
// retired interval_event_t::op_kind tag as the disambiguation check.
int
count_interval_resolutions(const profiler_hub::reader_t&                 r,
                           const profiler_hub::reader_types::event_id_t& id)
{
    int n = 0;
    n += r.get_region_details(id).has_value() ? 1 : 0;
    n += r.get_kernel_dispatch_details(id).has_value() ? 1 : 0;
    n += r.get_memory_copy_details(id).has_value() ? 1 : 0;
    n += r.get_memory_alloc_details(id).has_value() ? 1 : 0;
    return n;
}

// First track of a given type, or nullptr. Tests use this instead of
// hardcoding track ids so they stay robust to track-ordering / id-scheme
// differences between the v3 and v4 backends.
profiler_hub::reader_types::track_info_ptr_t
find_first_track(const profiler_hub::reader_types::track_info_list_t& tracks,
                 profiler_hub::reader_types::track_type_t             type)
{
    for(const auto& t : tracks)
    {
        if(t->type == type) return t;
    }
    return nullptr;
}

// All tracks of a given type.
profiler_hub::reader_types::track_info_list_t
find_tracks(const profiler_hub::reader_types::track_info_list_t& tracks,
            profiler_hub::reader_types::track_type_t             type)
{
    profiler_hub::reader_types::track_info_list_t out;
    for(const auto& t : tracks)
    {
        if(t->type == type) out.push_back(t);
    }
    return out;
}

// True if interval events are non-decreasing by start timestamp (the documented
// ordering contract of get_interval_track).
bool
is_start_sorted(const profiler_hub::reader_types::interval_event_list_t& v)
{
    for(size_t i = 1; i < v.size(); ++i)
    {
        if(v[i].start < v[i - 1].start) return false;
    }
    return true;
}

// True if scalar events are non-decreasing by timestamp (the documented
// ordering contract of get_scalar_track).
bool
is_timestamp_sorted(const profiler_hub::reader_types::scalar_event_list_t& v)
{
    for(size_t i = 1; i < v.size(); ++i)
    {
        if(v[i].timestamp < v[i - 1].timestamp) return false;
    }
    return true;
}

// Assert get_track_stats agrees with a full get_interval_track slice: count ==
// #rows, min_ts == MIN(start), max_ts == MAX(end). This is the core fidelity
// contract — the cheap aggregate must match what an eager load would compute.
void
expect_stats_match_intervals(
    const profiler_hub::reader_types::track_stats_t&         stats,
    const profiler_hub::reader_types::interval_event_list_t& intervals)
{
    ASSERT_EQ(stats.count, intervals.size());
    if(intervals.empty())
    {
        ASSERT_FALSE(stats.min_ts.has_value());
        ASSERT_FALSE(stats.max_ts.has_value());
        return;
    }
    auto min_start = intervals.front().start;
    auto max_end   = intervals.front().end;
    for(const auto& iv : intervals)
    {
        if(iv.start < min_start) min_start = iv.start;
        if(iv.end > max_end) max_end = iv.end;
    }
    ASSERT_TRUE(stats.min_ts.has_value());
    ASSERT_TRUE(stats.max_ts.has_value());
    ASSERT_EQ(stats.min_ts.value(), min_start);
    ASSERT_EQ(stats.max_ts.value(), max_end);
}

// Assert get_track_stats agrees with a full get_scalar_track slice: count ==
// #samples, min_ts == MIN(timestamp), max_ts == MAX(timestamp).
void
expect_stats_match_scalars(const profiler_hub::reader_types::track_stats_t&       stats,
                           const profiler_hub::reader_types::scalar_event_list_t& samples)
{
    ASSERT_EQ(stats.count, samples.size());
    if(samples.empty())
    {
        ASSERT_FALSE(stats.min_ts.has_value());
        ASSERT_FALSE(stats.max_ts.has_value());
        return;
    }
    auto min_ts = samples.front().timestamp;
    auto max_ts = samples.front().timestamp;
    for(const auto& s : samples)
    {
        if(s.timestamp < min_ts) min_ts = s.timestamp;
        if(s.timestamp > max_ts) max_ts = s.timestamp;
    }
    ASSERT_TRUE(stats.min_ts.has_value());
    ASSERT_TRUE(stats.max_ts.has_value());
    ASSERT_EQ(stats.min_ts.value(), min_ts);
    ASSERT_EQ(stats.max_ts.value(), max_ts);
}

class reader_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_test, create_reader_instance) { ASSERT_NE(m_reader, nullptr); }

TEST_F(reader_test, get_node_list_returns_correct_value)
{
    auto node_list = m_reader->get_all_nodes();
    ASSERT_EQ(node_list.size(), 1);

    ASSERT_EQ(node_list[0]->node_id, 9162464413581981795);
    ASSERT_EQ(node_list[0]->hash, 9162464413581981795);
    ASSERT_EQ(node_list[0]->machine_id, "7cd7e017ddf442f5b7ce8428af366498");
    ASSERT_EQ(node_list[0]->system_name, "Linux");
    ASSERT_EQ(node_list[0]->hostname, "smci350-zts-gtu-c14-05");
    ASSERT_EQ(node_list[0]->release, "5.15.0-70-generic");
    ASSERT_EQ(node_list[0]->version, "#77-Ubuntu SMP Tue Mar 21 14:02:37 UTC 2023");
    ASSERT_EQ(node_list[0]->hardware_name, "x86_64");
    ASSERT_EQ(node_list[0]->domain_name, "(none)");
}

TEST_F(reader_test, get_process_list_returns_correct_value)
{
    auto process_list = m_reader->get_all_processes();
    ASSERT_EQ(process_list.size(), 1);

    ASSERT_EQ(process_list[0]->pid, 67979);
    ASSERT_EQ(process_list[0]->ppid, 67166);
    ASSERT_EQ(process_list[0]->command, "./bit_extract");
    ASSERT_EQ(process_list[0]->node_info->node_id, 9162464413581981795);
}

TEST_F(reader_test, get_thread_list_returns_correct_value)
{
    auto thread_list = m_reader->get_all_threads();
    ASSERT_EQ(thread_list.size(), 4);

    // First thread
    ASSERT_EQ(thread_list[0]->thread_id, 67979);
    ASSERT_EQ(thread_list[0]->parent_process_id, 67166);
    ASSERT_EQ(thread_list[0]->name, "Thread 67979");
    ASSERT_EQ(thread_list[0]->start, 1702525691);
    ASSERT_EQ(thread_list[0]->process_info->pid, 67979);
    ASSERT_EQ(thread_list[0]->node_info->node_id, 9162464413581981795);

    // Second thread
    ASSERT_EQ(thread_list[1]->thread_id, 67991);
    ASSERT_EQ(thread_list[1]->name, "Thread 67991");
}

TEST_F(reader_test, get_agent_list_returns_correct_value)
{
    auto agent_list = m_reader->get_all_agents();
    ASSERT_EQ(agent_list.size(), 10);

    // Raw rocpd_info_agent.id is exposed so callers can key the agent (topology
    // nesting / cached-table lookups) without re-querying. Fixture ids run 1..10.
    ASSERT_EQ(agent_list[0]->id, 1);
    ASSERT_EQ(agent_list[2]->id, 3);

    ASSERT_EQ(agent_list[0]->agent_type, "CPU");
    ASSERT_EQ(agent_list[0]->type_index, 0);
    ASSERT_EQ(agent_list[0]->absolute_index, 0);
    ASSERT_EQ(agent_list[0]->logical_index, 0);
    ASSERT_EQ(agent_list[0]->name, "AMD EPYC 9575F 64-Core Processor");
    ASSERT_EQ(agent_list[0]->model_name, "");
    ASSERT_EQ(agent_list[0]->vendor_name, "CPU");
    ASSERT_EQ(agent_list[0]->product_name, "AMD EPYC 9575F 64-Core Processor");
    ASSERT_EQ(agent_list[0]->process_info->pid, 67979);
    ASSERT_EQ(agent_list[0]->node_info->node_id, 9162464413581981795);

    ASSERT_EQ(agent_list[2]->agent_type, "GPU");
    ASSERT_EQ(agent_list[2]->type_index, 0);
    ASSERT_EQ(agent_list[2]->absolute_index, 2);
    ASSERT_EQ(agent_list[2]->name, "gfx950");
    ASSERT_EQ(agent_list[2]->model_name, "ip discovery");
    ASSERT_EQ(agent_list[2]->vendor_name, "AMD");
    ASSERT_EQ(agent_list[2]->product_name, "AMD Instinct MI350X");
}

TEST_F(reader_test, get_stream_list_returns_correct_value)
{
    auto stream_list = m_reader->get_all_streams();
    ASSERT_EQ(stream_list.size(), 1);

    ASSERT_EQ(stream_list[0]->stream_id, 0);
    ASSERT_EQ(stream_list[0]->name, "Stream 0");
    ASSERT_EQ(stream_list[0]->process_info->pid, 67979);
    ASSERT_EQ(stream_list[0]->node_info->node_id, 9162464413581981795);
}

TEST_F(reader_test, get_queue_list_returns_correct_value)
{
    auto queue_list = m_reader->get_all_queues();
    ASSERT_EQ(queue_list.size(), 2);

    ASSERT_EQ(queue_list[0]->queue_id, 0);
    ASSERT_EQ(queue_list[0]->name, "Queue 0");
    ASSERT_EQ(queue_list[0]->process_info->pid, 67979);
    ASSERT_EQ(queue_list[0]->node_info->node_id, 9162464413581981795);

    ASSERT_EQ(queue_list[1]->queue_id, 1);
    ASSERT_EQ(queue_list[1]->name, "Queue 1");
}

TEST_F(reader_test, get_kernel_symbol_list_returns_correct_value)
{
    auto kernel_symbol_list = m_reader->get_all_kernel_symbols();
    ASSERT_EQ(kernel_symbol_list.size(), 11);

    // First kernel symbol
    ASSERT_EQ(kernel_symbol_list[0]->id, 1);
    ASSERT_EQ(kernel_symbol_list[0]->name, "__amd_rocclr_initHeap.kd");
    ASSERT_EQ(kernel_symbol_list[0]->display_name, "__amd_rocclr_initHeap.kd");
    ASSERT_EQ(kernel_symbol_list[0]->kernel_object, 2953328576);
    ASSERT_EQ(kernel_symbol_list[0]->kernarg_segment_size, 24);
    ASSERT_EQ(kernel_symbol_list[0]->kernarg_segment_alignment, 16);
    ASSERT_EQ(kernel_symbol_list[0]->sgpr_count, 32);
    ASSERT_EQ(kernel_symbol_list[0]->arch_vgpr_count, 8);
    ASSERT_EQ(kernel_symbol_list[0]->code_object_info->id, 1);
    ASSERT_EQ(kernel_symbol_list[0]->process_info->pid, 67979);
    ASSERT_EQ(kernel_symbol_list[0]->node_info->node_id, 9162464413581981795);
}

TEST_F(reader_test, get_code_object_list_returns_correct_value)
{
    auto code_object_list = m_reader->get_all_code_objects();
    ASSERT_EQ(code_object_list.size(), 2);

    // First code object
    ASSERT_EQ(code_object_list[0]->id, 1);
    ASSERT_EQ(code_object_list[0]->uri, "memory://67979#offset=0x4608f10&size=32640");
    ASSERT_EQ(code_object_list[0]->load_base, 140018887163904);
    ASSERT_EQ(code_object_list[0]->load_size, 36864);
    ASSERT_EQ(code_object_list[0]->load_delta, 140018887163904);
    ASSERT_EQ(code_object_list[0]->storage_type, "MEMORY");
    ASSERT_EQ(code_object_list[0]->process_info->pid, 67979);
    ASSERT_EQ(code_object_list[0]->node_info->node_id, 9162464413581981795);
    ASSERT_EQ(code_object_list[0]->agent_info->agent_type, "GPU");
    ASSERT_EQ(code_object_list[0]->agent_info->type_index, 0);
}

TEST_F(reader_test, get_track_list_returns_correct_count)
{
    auto track_list = m_reader->get_all_tracks();
    // cpu_thread/region tracks are synthesized from rocpd_region, NOT read from the
    // 2369-row rocpd_track grab-bag. rocpd_track contributes only its 54 counter tracks
    // (rows referenced by rocpd_sample). Synthesis adds 1 gpu_queue + 2 dma + 1 region
    // (the sole (nid,pid,tid)=(...,67979,1) thread, all regions main => one track) +
    // 1 stream (the sole stream_id=0, aggregating kernel_dispatch + memory_copy) +
    // 1 memory (the sole rocpd_memory_allocate row keyed
    // (nid,agent_id=NULL,queue_id=NULL,pid)). dma tracks are keyed by destination agent
    // (nid,pid,queue_id,dst_agent_id): the two memory copies target dst_agent_id 1 and 3
    // => 2 dma tracks (was 1 when keyed by the shared stream_id=0):
    //   54 counter + 1 gpu_queue + 2 dma + 1 cpu_thread + 1 stream + 1 memory = 60.
    // Task 012B adds memory_activity tracks: 1 per distinct non-null (nid, pid, agent_id)
    // in rocpd_memory_allocate. The main fixture's sole allocate row has agent_id=NULL
    // (a FREE-recovery row), so it does not form a memory_activity track => still 60.
    ASSERT_EQ(track_list.size(), 60);
}

TEST_F(reader_test, get_track_list_first_track_has_correct_values)
{
    auto track_list = m_reader->get_all_tracks();
    ASSERT_GE(track_list.size(), 1);

    // The real capture has exactly one synthesized cpu_thread (region) track, for the
    // sole region-bearing thread (nid,pid,tid)=(...,67979,1). Its identity resolves
    // through the info tables and its name comes from rocpd_info_thread.name.
    auto cpu =
        find_tracks(track_list, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_EQ(cpu.size(), 1U);
    const auto& t = cpu.front();
    ASSERT_EQ(t->name, "Thread 67979");
    ASSERT_EQ(t->region_kind, profiler_hub::reader_types::region_track_kind_t::main);
    ASSERT_NE(t->node_info, nullptr);
    ASSERT_EQ(t->node_info->node_id, 9162464413581981795);
    ASSERT_NE(t->process_info, nullptr);
    ASSERT_EQ(t->process_info->pid, 67979);
    ASSERT_NE(t->thread_info, nullptr);
    ASSERT_EQ(t->thread_info->thread_id, 67979);
}

TEST_F(reader_test, get_pmc_info_list_returns_correct_count)
{
    auto pmc_list = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_list.size(), 2358);
}

TEST_F(reader_test, get_pmc_info_list_first_item_has_correct_values)
{
    auto pmc_list = m_reader->get_all_pmc_info();
    ASSERT_GE(pmc_list.size(), 1);

    // First PMC info
    ASSERT_EQ(pmc_list[0]->name, "device_jpeg_activity_5_28");
    ASSERT_EQ(pmc_list[0]->agent_info->agent_type, "GPU");
    ASSERT_EQ(pmc_list[0]->target_arch, "GPU");
    ASSERT_EQ(pmc_list[0]->symbol, "JpegAct_5_28");
    ASSERT_EQ(pmc_list[0]->description, "JPEG Activity of a GPU device");
    ASSERT_EQ(pmc_list[0]->units, "%");
    ASSERT_EQ(pmc_list[0]->value_type, "ABS");
    ASSERT_EQ(pmc_list[0]->is_constant, 0);
    ASSERT_EQ(pmc_list[0]->is_derived, 0);
    ASSERT_EQ(pmc_list[0]->process_info->pid, 67979);
    ASSERT_EQ(pmc_list[0]->node_info->node_id, 9162464413581981795);
}

TEST_F(reader_test, get_events_returns_non_empty_list)
{
    auto events = m_reader->get_events();
    ASSERT_GT(events.size(), 0);
}

TEST_F(reader_test, get_events_with_type_filter_region)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types = { profiler_hub::reader_types::event_type_t::region };
    auto events  = m_reader->get_events(filter);
    ASSERT_GT(events.size(), 0);

    for(const auto& event : events)
    {
        ASSERT_EQ(event.unique_identifier.type,
                  profiler_hub::reader_types::event_type_t::region);
    }
}

TEST_F(reader_test, get_events_region_has_correct_fields)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::region };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    const auto& event = events[0];
    ASSERT_EQ(event.unique_identifier.type,
              profiler_hub::reader_types::event_type_t::region);
    ASSERT_GT(event.unique_identifier.id, 0);
    ASSERT_GT(event.start_timestamp, 0);
    ASSERT_GE(event.end_timestamp, event.start_timestamp);
    ASSERT_FALSE(event.display_name.empty());
}

TEST_F(reader_test, get_events_with_pagination_limit)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.pagination = { 5, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_LE(events.size(), 5);
    ASSERT_GT(events.size(), 0);
}

TEST_F(reader_test, get_events_with_pagination_offset)
{
    auto all_events = m_reader->get_events();

    profiler_hub::reader_types::event_filter_t filter;
    filter.pagination  = { std::nullopt, 2 };
    auto offset_events = m_reader->get_events(filter);

    ASSERT_EQ(offset_events.size(), all_events.size() - 2);
}

TEST_F(reader_test, get_events_for_track_returns_events)
{
    auto tracks = m_reader->get_all_tracks();
    ASSERT_GT(tracks.size(), 0);

    bool found_events = false;
    for(const auto& track : tracks)
    {
        auto events = m_reader->get_events_for_track(track);
        if(!events.empty())
        {
            found_events = true;
            for(const auto& event : events)
            {
                ASSERT_NE(event.track, nullptr);
            }
            break;
        }
    }
    ASSERT_TRUE(found_events);
}

TEST_F(reader_test, get_event_count_matches_events_size)
{
    auto count  = m_reader->get_event_count();
    auto events = m_reader->get_events();
    ASSERT_EQ(count, events.size());
}

TEST_F(reader_test, get_event_count_ignores_pagination)
{
    const auto total = m_reader->get_event_count();
    ASSERT_GT(total, 1U);

    profiler_hub::reader_types::event_filter_t paged_filter;
    paged_filter.pagination.limit  = 1;
    paged_filter.pagination.offset = 0;

    ASSERT_EQ(m_reader->get_event_count(paged_filter), total);
}

TEST_F(reader_test, get_event_count_respects_types_filter)
{
    const auto counts = m_reader->get_event_counts({});

    profiler_hub::reader_types::event_filter_t region_filter;
    region_filter.types = { profiler_hub::reader_types::event_type_t::region };
    ASSERT_EQ(m_reader->get_event_count(region_filter),
              counts.at(profiler_hub::reader_types::event_type_t::region));

    profiler_hub::reader_types::event_filter_t dispatch_filter;
    dispatch_filter.types = { profiler_hub::reader_types::event_type_t::kernel_dispatch };
    ASSERT_EQ(m_reader->get_event_count(dispatch_filter),
              counts.at(profiler_hub::reader_types::event_type_t::kernel_dispatch));
}

TEST_F(reader_test, get_event_count_with_time_window_matches_filtered_events)
{
    const auto unfiltered = m_reader->get_events();
    ASSERT_FALSE(unfiltered.empty());

    auto first_start = unfiltered.front().start_timestamp;
    auto last_start  = unfiltered.back().start_timestamp;
    if(last_start < first_start) std::swap(first_start, last_start);

    const auto mid = first_start + (last_start - first_start) / 2;

    profiler_hub::reader_types::event_filter_t windowed;
    windowed.time_window.start = first_start;
    windowed.time_window.end   = mid;

    const auto windowed_events = m_reader->get_events(windowed);
    ASSERT_EQ(m_reader->get_event_count(windowed), windowed_events.size());
}

// ============================================================================
// Event detail tests
// ============================================================================

TEST_F(reader_test, get_region_details_first_region_has_correct_values)
{
    // First region in DB: id=1, start=23040314699996, end=23040314726875, name="mbind"
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::region };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_region_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->start_timestamp, 23040314699996);
    ASSERT_EQ(details->end_timestamp, 23040314726875);
    ASSERT_EQ(details->name, "mbind");
}

TEST_F(reader_test, get_region_details_has_event_metadata)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::region };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_region_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_NE(details->event, nullptr);
    // First region event_id=28 has category_id=3302 -> "numa"
    ASSERT_EQ(details->event->event_category, "numa");
}

TEST_F(reader_test, get_region_details_returns_nullopt_for_wrong_type)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::kernel_dispatch };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_region_details(events[0]);
    ASSERT_FALSE(details.has_value());
}

TEST_F(reader_test, get_kernel_dispatch_details_has_correct_values)
{
    // DB has 1 kernel dispatch: id=1, dispatch_id=1, wg=256x1x1, grid=131072x1x1
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::kernel_dispatch };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_kernel_dispatch_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->dispatch_id, 1);
    ASSERT_EQ(details->start_timestamp, 23040497580868);
    ASSERT_EQ(details->end_timestamp, 23040497591788);
    ASSERT_EQ(details->workgroup_size_x, 256);
    ASSERT_EQ(details->workgroup_size_y, 1);
    ASSERT_EQ(details->workgroup_size_z, 1);
    ASSERT_EQ(details->grid_size_x, 131072);
    ASSERT_EQ(details->grid_size_y, 1);
    ASSERT_EQ(details->grid_size_z, 1);
}

TEST_F(reader_test, get_kernel_dispatch_details_resolves_kernel_symbol)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::kernel_dispatch };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_kernel_dispatch_details(events[0]);
    ASSERT_TRUE(details.has_value());
    // kernel_id=11 -> display_name contains "bit_extract_kernel"
    ASSERT_NE(details->kernel_symbol_info, nullptr);
    EXPECT_NE(details->kernel_symbol_info->display_name.find("bit_extract_kernel"),
              std::string::npos);
}

TEST_F(reader_test, get_kernel_dispatch_details_resolves_node_and_process)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::kernel_dispatch };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_kernel_dispatch_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_NE(details->node_info, nullptr);
    ASSERT_EQ(details->node_info->node_id, 9162464413581981795);
    ASSERT_NE(details->process_info, nullptr);
    ASSERT_EQ(details->process_info->pid, 67979);
}

TEST_F(reader_test, get_memory_copy_details_has_correct_values)
{
    // DB has 2 memory copies. First: id=1, size=4000000, name=MEMORY_COPY_HOST_TO_DEVICE
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::memory_copy };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_memory_copy_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->start_timestamp, 23040496787705);
    ASSERT_EQ(details->end_timestamp, 23040496865705);
    ASSERT_EQ(details->size, 4000000);
    ASSERT_EQ(details->name, "MEMORY_COPY_HOST_TO_DEVICE");
}

TEST_F(reader_test, get_memory_copy_details_resolves_agents)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::memory_copy };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_memory_copy_details(events[0]);
    ASSERT_TRUE(details.has_value());
    // dst_agent_id=3, src_agent_id=1
    ASSERT_NE(details->dst_agent_id, nullptr);
    ASSERT_NE(details->src_agent_id, nullptr);
}

TEST_F(reader_test, get_memory_alloc_details_has_correct_values)
{
    // Inserted test data: id=1, type=ALLOC, level=REAL, size=4096, address=1048576
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::memory_allocate };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_memory_alloc_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->type, "ALLOC");
    ASSERT_EQ(details->level, "REAL");
    ASSERT_EQ(details->start_timestamp, 23040314700000);
    ASSERT_EQ(details->end_timestamp, 23040314710000);
    ASSERT_TRUE(details->address.has_value());
    ASSERT_EQ(details->address.value(), 1048576);
    ASSERT_EQ(details->size, 4096);
}

TEST_F(reader_test, get_memory_alloc_details_has_event_with_call_stack)
{
    // The inserted memory_allocate event has call_stack JSON with hipMalloc
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::memory_allocate };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto details = m_reader->get_memory_alloc_details(events[0]);
    ASSERT_TRUE(details.has_value());
    ASSERT_NE(details->event, nullptr);

    // call_stack should be deserialized from JSON
    ASSERT_FALSE(details->event->call_stack.empty());
    ASSERT_TRUE(details->event->call_stack.front().program_counter.has_value());
    ASSERT_EQ(details->event->call_stack.front().program_counter->function, "hipMalloc");
    ASSERT_EQ(details->event->call_stack.front().program_counter->filename,
              "/opt/rocm/hip/src/hip_memory.cpp");
    ASSERT_TRUE(
        details->event->call_stack.front().program_counter->line_number.has_value());
    ASSERT_EQ(details->event->call_stack.front().program_counter->line_number.value(),
              123);

    // line_info should also be deserialized
    ASSERT_FALSE(details->event->line_info_list.empty());
    ASSERT_TRUE(details->event->line_info_list.front().program_counter.has_value());
    ASSERT_EQ(details->event->line_info_list.front().program_counter->function,
              "hipMalloc");
}

// ============================================================================
// Event property tests
// ============================================================================

TEST_F(reader_test, get_call_stack_for_memory_alloc_returns_hipMalloc)
{
    // The inserted memory_allocate event has call_stack with hipMalloc
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::memory_allocate };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto stack = m_reader->get_call_stack(events[0]);
    ASSERT_EQ(stack.size(), 1);
    ASSERT_TRUE(stack.front().program_counter.has_value());
    ASSERT_EQ(stack.front().program_counter->function, "hipMalloc");

    ASSERT_TRUE(stack.front().address_range.has_value());
    ASSERT_EQ(stack.front().address_range->address_base, 4096);
    ASSERT_EQ(stack.front().address_range->address_high, 8192);
}

TEST_F(reader_test, get_source_context_for_memory_alloc_returns_entry)
{
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::memory_allocate };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto context = m_reader->get_source_context(events[0]);
    ASSERT_EQ(context.size(), 1);
    ASSERT_TRUE(context.front().program_counter.has_value());
    ASSERT_EQ(context.front().program_counter->function, "hipMalloc");
}

TEST_F(reader_test, get_call_stack_returns_empty_for_no_call_stack)
{
    // Region events in this DB have empty call_stack JSON
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::region };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto stack = m_reader->get_call_stack(events[0]);
    ASSERT_TRUE(stack.empty());
}

TEST_F(reader_test, get_arguments_for_hipGetDevice_has_correct_values)
{
    // Region id=23 (hipGetDevice, event_id=86) has 1 arg: pos=0, type=int*, name=deviceId
    // Region id=22 (hipGetDevice, event_id=85) has 0 args
    // Find the hipGetDevice instance that has args and verify values
    profiler_hub::reader_types::event_filter_t filter;
    filter.types = { profiler_hub::reader_types::event_type_t::region };
    auto events  = m_reader->get_events(filter);
    ASSERT_GT(events.size(), 0);

    bool found = false;
    for(const auto& event : events)
    {
        if(event.display_name != "hipGetDevice") continue;

        auto args = m_reader->get_arguments(event);
        if(args.empty()) continue;

        ASSERT_EQ(args.size(), 1);
        ASSERT_EQ(args[0]->position, 0);
        ASSERT_EQ(args[0]->type, "int*");
        ASSERT_EQ(args[0]->name, "deviceId");
        ASSERT_EQ(args[0]->value, "0");
        found = true;
        break;
    }
    ASSERT_TRUE(found) << "No hipGetDevice region with arguments found";
}

TEST_F(reader_test, get_arguments_returns_empty_for_event_without_args)
{
    // First region (mbind) has event_id=28 with 0 args
    profiler_hub::reader_types::event_filter_t filter;
    filter.types      = { profiler_hub::reader_types::event_type_t::region };
    filter.pagination = { 1, std::nullopt };
    auto events       = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto args = m_reader->get_arguments(events[0]);
    ASSERT_TRUE(args.empty());
}

TEST_F(reader_test, get_correlated_events_finds_related_events)
{
    // stack_id=7 has 2 events (event_id 182 and 203).
    // event_id=203 is in memory_copy (MC id=1)
    // We need to find the memory_copy event, then check its correlated events
    profiler_hub::reader_types::event_filter_t filter;
    filter.types = { profiler_hub::reader_types::event_type_t::memory_copy };
    auto events  = m_reader->get_events(filter);
    ASSERT_GE(events.size(), 1);

    auto correlated = m_reader->get_correlated_events(events[0]);
    // Should find at least 1 correlated event (the region with the same stack_id)
    ASSERT_GE(correlated.size(), 1);
    // Correlated events should have valid IDs and not be the same event
    for(const auto& ce : correlated)
    {
        ASSERT_GT(ce.unique_identifier.id, 0);
    }
}

// ============================================================================
// Database metadata tests
// ============================================================================

TEST_F(reader_test, get_data_time_range_has_correct_values)
{
    auto range = m_reader->get_data_time_range();
    ASSERT_TRUE(range.start.has_value());
    ASSERT_TRUE(range.end.has_value());
    // min across all tables: 23040260707644, max: 23040498732102
    ASSERT_EQ(range.start.value(), 23040260707644);
    ASSERT_EQ(range.end.value(), 23040498732102);
}

TEST_F(reader_test, get_event_counts_has_correct_values)
{
    auto counts = m_reader->get_event_counts();

    // DB: 59 regions, 1 kernel dispatch, 2 memory copies, 1 memory allocate
    auto region_it = counts.find(profiler_hub::reader_types::event_type_t::region);
    ASSERT_NE(region_it, counts.end());
    ASSERT_EQ(region_it->second, 59);

    auto kd_it = counts.find(profiler_hub::reader_types::event_type_t::kernel_dispatch);
    ASSERT_NE(kd_it, counts.end());
    ASSERT_EQ(kd_it->second, 1);

    auto mc_it = counts.find(profiler_hub::reader_types::event_type_t::memory_copy);
    ASSERT_NE(mc_it, counts.end());
    ASSERT_EQ(mc_it->second, 2);

    auto ma_it = counts.find(profiler_hub::reader_types::event_type_t::memory_allocate);
    ASSERT_NE(ma_it, counts.end());
    ASSERT_EQ(ma_it->second, 1);
}

TEST_F(reader_test, get_event_counts_total_matches_get_events)
{
    auto counts = m_reader->get_event_counts();
    auto events = m_reader->get_events();

    size_t total = 0;
    for(const auto& [type, count] : counts)
    {
        total += count;
    }
    ASSERT_EQ(total, events.size());
}

// ============================================================================
// Track-scoped API tests — v3 (rocpd.db)
// get_interval_track / get_scalar_track / get_flows / track_info_t
// ============================================================================

TEST_F(reader_test, v3_tracks_have_types_and_core_identity)
{
    auto tracks = m_reader->get_all_tracks();
    ASSERT_FALSE(tracks.empty());

    // Every track carries the always-populated identity anchors.
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->node_info, nullptr);
        ASSERT_NE(t->process_info, nullptr);
    }

    // The v3 fixture exercises both an interval (cpu_thread) and a scalar
    // (counter) track type.
    ASSERT_NE(
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread),
        nullptr);
    ASSERT_NE(find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter),
              nullptr);
}

TEST_F(reader_test, v3_get_interval_track_cpu_thread_ordered_values)
{
    auto tracks = m_reader->get_all_tracks();
    auto cpu_tracks =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_FALSE(cpu_tracks.empty());

    // Exactly one cpu_thread track carries the 59 region events (all on tid=1).
    profiler_hub::reader_types::interval_event_list_t region_intervals;
    for(const auto& t : cpu_tracks)
    {
        auto intervals = m_reader->get_interval_track(t->id);
        if(intervals.size() == 59)
        {
            region_intervals = std::move(intervals);
            break;
        }
    }
    ASSERT_EQ(region_intervals.size(), 59)
        << "no cpu_thread track returned the expected 59 region intervals";

    // Ordered by start ascending.
    ASSERT_TRUE(is_start_sorted(region_intervals));

    // First interval (region id=59) has known start/end and resolvable details.
    const auto& first = region_intervals.front();
    ASSERT_EQ(row_id_of(first.id), 59U);
    ASSERT_EQ(first.start, 23040260707644);
    ASSERT_EQ(first.end, 23040498732102);
    ASSERT_GE(first.end, first.start);

    auto details = m_reader->get_region_details(first.id);
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->name, "bit_extract");
}

TEST_F(reader_test, v3_get_interval_track_cpu_thread_carries_category)
{
    // Category is per-EVENT, not derivable from the track type or region kind: the
    // 59 regions on this one cpu_thread carry several distinct categories. The reader
    // resolves it via rocpd_string on the v3 backend; assert it round-trips against
    // the authoritative get_region_details() -> event->event_category oracle.
    auto tracks = m_reader->get_all_tracks();
    auto cpu_tracks =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_FALSE(cpu_tracks.empty());

    profiler_hub::reader_types::interval_event_list_t region_intervals;
    for(const auto& t : cpu_tracks)
    {
        auto intervals = m_reader->get_interval_track(t->id);
        if(intervals.size() == 59)
        {
            region_intervals = std::move(intervals);
            break;
        }
    }
    ASSERT_EQ(region_intervals.size(), 59);

    // First interval (region id=59) resolves to the "host" category.
    ASSERT_EQ(row_id_of(region_intervals.front().id), 59U);
    ASSERT_EQ(region_intervals.front().category, "host");

    // Every interval's carried category matches the detail-path oracle, and the
    // track spans multiple distinct categories (host/numa/pthread/rocm_hip_api/
    // rocm_marker_api) -- proving fidelity is per-event, not per-track.
    std::set<std::string> seen;
    for(const auto& ev : region_intervals)
    {
        auto details = m_reader->get_region_details(ev.id);
        ASSERT_TRUE(details.has_value());
        ASSERT_NE(details->event, nullptr);
        ASSERT_EQ(ev.category, details->event->event_category);
        seen.insert(ev.category);
    }
    ASSERT_GT(seen.size(), 1U) << "expected several distinct per-event categories";
    ASSERT_TRUE(seen.count("host"));
    ASSERT_TRUE(seen.count("rocm_hip_api"));
    ASSERT_TRUE(seen.count("numa"));
}

TEST_F(reader_test, v3_gpu_queue_track_carries_agent_id)
{
    // The gpu_queue track exposes its owning agent's raw rocpd_info_agent.id via
    // agent_info->id (the same shared agent_info the reader caches). Callers need
    // this numeric id to nest the queue under its GPU in the topology view and to
    // key the "Agent" cached table -- neither is reachable without the raw id.
    auto tracks = m_reader->get_all_tracks();
    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);
    ASSERT_NE(gpu->agent_info, nullptr);
    // Fixture: the sole gpu_queue belongs to agent_id=3 (a GPU, absolute_index 2).
    ASSERT_EQ(gpu->agent_info->id, 3);
    ASSERT_EQ(gpu->agent_info->agent_type, "GPU");
}

TEST_F(reader_test, v3_get_interval_track_gpu_queue_carries_category)
{
    // gpu_queue kernel-dispatch intervals carry per-event category, resolved in-SQL
    // via rocpd_string on the v3 backend (LEFT JOIN, additive). Assert it round-trips
    // against the authoritative get_kernel_dispatch_details() -> event->event_category
    // oracle -- the same fidelity contract the region/stream interval tracks meet.
    auto tracks = m_reader->get_all_tracks();
    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);

    auto intervals = m_reader->get_interval_track(gpu->id);
    // Fixture: one kernel dispatch on this queue (agent_id=3, queue_id=1).
    ASSERT_EQ(intervals.size(), 1U);

    const auto& ev = intervals.front();
    ASSERT_EQ(ev.category, "rocm_kernel_dispatch");

    auto details = m_reader->get_kernel_dispatch_details(ev.id);
    ASSERT_TRUE(details.has_value());
    ASSERT_NE(details->event, nullptr);
    ASSERT_EQ(ev.category, details->event->event_category);
}

TEST_F(reader_test, v3_get_interval_track_dma_carries_category)
{
    // Standalone queue-keyed dma (memory-copy) intervals carry per-event category,
    // resolved in-SQL via rocpd_string on the v3 backend (LEFT JOIN, additive). Assert
    // it round-trips against the authoritative get_memory_copy_details() ->
    // event->event_category oracle -- the same fidelity contract region/gpu_queue meet.
    auto tracks = m_reader->get_all_tracks();
    auto dma    = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    // Fixture: 2 memory copies keyed by destination agent (queue_id=0, dst_agent_id 1 &
    // 3)
    // => 2 dma tracks, one copy each.
    ASSERT_EQ(dma.size(), 2U);

    size_t total = 0;
    for(const auto& track : dma)
    {
        auto intervals = m_reader->get_interval_track(track->id);
        total += intervals.size();
        for(const auto& ev : intervals)
        {
            ASSERT_EQ(ev.category, "rocm_memory_copy");
            auto details = m_reader->get_memory_copy_details(ev.id);
            ASSERT_TRUE(details.has_value());
            ASSERT_NE(details->event, nullptr);
            ASSERT_EQ(ev.category, details->event->event_category);
        }
    }
    ASSERT_EQ(total, 2U);
}

TEST_F(reader_test, v3_get_scalar_track_counter_ordered_and_details)
{
    auto tracks = m_reader->get_all_tracks();
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);

    auto samples = m_reader->get_scalar_track(counter->id);
    ASSERT_FALSE(samples.empty());
    ASSERT_TRUE(is_timestamp_sorted(samples));

    // First sample's value is reproducible via get_scalar_details() on its handle.
    const auto& first   = samples.front();
    auto        details = m_reader->get_scalar_details(first.id);
    ASSERT_TRUE(details.has_value());
    ASSERT_DOUBLE_EQ(details->value, first.value);
    ASSERT_EQ(details->sample.timestamp, first.timestamp);
    ASSERT_NE(details->sample.track, nullptr);
}

TEST_F(reader_test, v3_counter_track_has_no_agent_info)
{
    // Q10: v3 rocpd_track has no agent_id column, so a v3 counter track can NEVER
    // carry agent_info (differs from v4). This is a schema invariant, not a
    // property of this capture.
    //
    // thread_info, by contrast, is driven purely by rocpd_track.tid (nullable) and
    // is orthogonal to counter classification (reader_impl.cpp populates it from
    // tid regardless of type). In this bundled capture every counter track has
    // tid=NULL, so thread_info is null here. The tid-present branch (a v3 counter
    // that DOES carry a thread) is covered by reader_v3_edge_test below.
    auto tracks = m_reader->get_all_tracks();
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);
    ASSERT_EQ(counter->agent_info, nullptr);
    ASSERT_EQ(counter->thread_info, nullptr);
}

TEST_F(reader_test, v3_counter_tracks_resolve_deterministic_pmc)
{
    // Regression: 005B-4-fix-1-fix-1. One AMD-SMI poll co-samples all of an agent's
    // metrics under a single rocpd_sample.event_id, so a plain sample->pmc_event join
    // on event_id fans each of the 54 counter tracks out to 6 candidate pmc_ids; a bare
    // GROUP BY track_id then keeps an arbitrary one -- e.g. giving device_busy_mm [0]
    // the device_busy_gfx pmc. Each track must instead resolve to the ONE pmc that
    // matches its own metric identity, and its Q9 display name must be that pmc's name.
    //
    // Ground truth (verified against tests/unit/rocpd.db): rocpd_track.name_id encodes
    // "<metric> [<ordinal>]" for the 48 device tracks -- the ordinal equals the GPU
    // agent type_index -- and a bare "<metric>" for the 6 process tracks (CPU agent,
    // type_index 0). Track ids are stable in this committed fixture.
    struct expected_t
    {
        std::string metric;
        std::string agent_type;
        size_t      type_index;
    };

    std::map<size_t, expected_t> expected;
    const char*                  device_metrics[] = { "device_busy_gfx", "device_busy_mm",
                                                      "device_busy_umc", "device_memory_usage",
                                                      "device_power",    "device_temp" };
    const size_t                 device_bases[]   = { 12, 20, 28, 2084, 2092, 2100 };
    for(size_t m = 0; m < 6; ++m)
    {
        for(size_t ord = 0; ord < 8; ++ord)
        {
            expected[device_bases[m] + ord] = expected_t{ device_metrics[m], "GPU", ord };
        }
    }
    expected[2364] = { "process_context_switch", "CPU", 0 };
    expected[2365] = { "process_kernel_cpu_time", "CPU", 0 };
    expected[2366] = { "process_memory_hwm", "CPU", 0 };
    expected[2367] = { "process_page_fault", "CPU", 0 };
    expected[2368] = { "process_user_cpu_time", "CPU", 0 };
    expected[2369] = { "process_virtual_memory", "CPU", 0 };

    auto tracks = m_reader->get_all_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_EQ(counters.size(), 54U);

    std::set<std::string> resolved_identities;
    for(const auto& t : counters)
    {
        auto it = expected.find(t->id);
        ASSERT_NE(it, expected.end()) << "unexpected counter track id " << t->id;
        const auto& exp = it->second;

        // The fix attaches the deterministically-resolved pmc panel to each track...
        ASSERT_NE(t->pmc_info, nullptr) << "track " << t->id << " missing pmc_info";
        // ...and (005B-4-fix-1-fix-2) exposes that pmc's numeric id on every counter
        // track.
        ASSERT_NE(t->pmc_info->pmc_id, 0U)
            << "track " << t->id << " missing numeric pmc_id";
        ASSERT_EQ(t->pmc_info->name, exp.metric) << "track " << t->id;
        // ...and corrects the Q9 display name to that same pmc's name (previously the
        // arbitrary fanned name, wrong on 45 of 54 tracks).
        ASSERT_EQ(t->name, t->pmc_info->name) << "track " << t->id;
        // Agent scoping: the resolved pmc belongs to the agent the track name names.
        ASSERT_NE(t->pmc_info->agent_info, nullptr) << "track " << t->id;
        ASSERT_EQ(t->pmc_info->agent_info->agent_type, exp.agent_type)
            << "track " << t->id;
        ASSERT_EQ(t->pmc_info->agent_info->type_index, exp.type_index)
            << "track " << t->id;

        // Each resolved (metric, agent) identity must be unique across the 54 tracks --
        // proves the true 1:1 track<->pmc mapping, not an arbitrary fan-out duplicate.
        std::string identity = t->pmc_info->name + "/" +
                               t->pmc_info->agent_info->agent_type + "/" +
                               std::to_string(t->pmc_info->agent_info->type_index);
        ASSERT_TRUE(resolved_identities.insert(identity).second)
            << "duplicate resolved identity: " << identity;
    }
    ASSERT_EQ(resolved_identities.size(), 54U);
}

TEST_F(reader_test, v3_scalar_value_query_strips_pmc_fanout)
{
    // Regression: 005B-4-fix-3. 005B-4-fix-1-fix-1 fixed the counter *metadata* query so
    // each track's name/pmc_info resolve to its own pmc. The four *value/detail* queries
    // (scalar_track / scalar_stats / scalar_detail / pmc_event_detail) still used the
    // naive sample->pmc_event join on the shared event_id, so a track's values were
    // fanned out to ALL co-sampled pmcs under each poll. On tests/unit/rocpd.db the
    // device_busy_gfx [0] track (16 samples) returned 96 scalar rows mixing six metrics.
    // The resolved_pmc_join must collapse it back to exactly the track's own 16 samples.
    auto tracks = m_reader->get_all_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);

    // Resolve the target track by its (metric, agent) identity, not a hard-coded id.
    profiler_hub::reader_types::track_info_ptr_t gfx0;
    for(const auto& t : counters)
    {
        if(t->pmc_info != nullptr && t->pmc_info->name == "device_busy_gfx" &&
           t->pmc_info->agent_info != nullptr && t->pmc_info->agent_info->type_index == 0)
        {
            gfx0 = t;
            break;
        }
    }
    ASSERT_NE(gfx0, nullptr) << "device_busy_gfx [0] counter track not found";

    // 005B-4-fix-1-fix-2: gfx0 is track 12 in this committed fixture; its exposed numeric
    // pmc_id must be the resolver's rn=1 pick (pmc 1796), not 0 or a fanned-out
    // neighbour.
    EXPECT_EQ(gfx0->id, 12U);
    ASSERT_NE(gfx0->pmc_info, nullptr);
    EXPECT_EQ(gfx0->pmc_info->pmc_id, 1796U);

    auto samples = m_reader->get_scalar_track(gfx0->id);
    ASSERT_EQ(samples.size(), 16U)
        << "fan-out not stripped (expected 16, pre-fix was 96)";
    ASSERT_TRUE(is_timestamp_sorted(samples));

    // scalar_stats must agree with the de-fanned scalar_track slice (both now
    // resolver-joined).
    auto stats = m_reader->get_track_stats(gfx0->id);
    expect_stats_match_scalars(stats, samples);
    ASSERT_EQ(stats.count, 16U);

    // Every sample's value is reproducible via get_scalar_details (scalar_detail, also
    // resolver-joined) -- i.e. each opaque id resolves to the track's own single pmc
    // value, not one of the six fanned metrics.
    for(const auto& s : samples)
    {
        auto details = m_reader->get_scalar_details(s.id);
        ASSERT_TRUE(details.has_value()) << "sample row " << row_id_of(s.id);
        ASSERT_DOUBLE_EQ(details->value, s.value);
        ASSERT_EQ(details->sample.timestamp, s.timestamp);
    }
}

TEST_F(reader_test, v3_get_interval_track_on_counter_returns_empty)
{
    // Q7: an interval query against a counter (scalar-only) track returns empty.
    auto tracks = m_reader->get_all_tracks();
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);
    ASSERT_TRUE(m_reader->get_interval_track(counter->id).empty());
}

TEST_F(reader_test, v3_get_scalar_track_on_cpu_thread_returns_empty)
{
    // Q7: a scalar query against a non-counter track returns empty.
    auto tracks = m_reader->get_all_tracks();
    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);
    ASSERT_TRUE(m_reader->get_scalar_track(cpu->id).empty());
}

TEST_F(reader_test, v3_track_scoped_queries_unknown_id_return_empty)
{
    // Unknown track id is not an error; both accessors return empty.
    constexpr size_t kUnknownTrackId = 999999999;
    ASSERT_TRUE(m_reader->get_interval_track(kUnknownTrackId).empty());
    ASSERT_TRUE(m_reader->get_scalar_track(kUnknownTrackId).empty());
}

TEST_F(reader_test, v3_get_flows_links_regions_to_gpu_events)
{
    // v3 fixture flows: 1 region->kernel_dispatch + 2 region->memory_copy
    // + 0 region->memory_allocate = 3 total (stack_id linkage). This capture is a
    // flat clique (each stack has one region + one GPU event), so the new
    // region->region / sibling categories add nothing here; only region sources.
    auto flows = m_reader->get_flows();
    ASSERT_EQ(flows.size(), 3);
    for(const auto& f : flows)
    {
        // Endpoint type is now encoded in the opaque handle; a region source must
        // resolve through get_region_details() and each dest through exactly one of the
        // GPU-side detail accessors.
        ASSERT_GT(row_id_of(f.source), 0U);
        ASSERT_GT(row_id_of(f.dest), 0U);
        ASSERT_TRUE(m_reader->get_region_details(f.source).has_value());
        ASSERT_EQ(count_interval_resolutions(*m_reader, f.dest), 1);
        ASSERT_FALSE(m_reader->get_region_details(f.dest).has_value());
    }
}

TEST_F(reader_test, v3_get_track_stats_cpu_thread_matches_interval_slice)
{
    // The cpu_thread track carrying the 59 region events: stats must agree with the
    // full get_interval_track slice (count 59, min start, max end) without loading it.
    auto tracks = m_reader->get_all_tracks();
    auto cpu_tracks =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_FALSE(cpu_tracks.empty());

    bool checked = false;
    for(const auto& t : cpu_tracks)
    {
        auto intervals = m_reader->get_interval_track(t->id);
        if(intervals.size() != 59) continue;
        auto stats = m_reader->get_track_stats(t->id);
        expect_stats_match_intervals(stats, intervals);
        // Known absolute bounds: first region start (from the interval test above).
        ASSERT_EQ(stats.count, 59U);
        ASSERT_EQ(stats.min_ts.value(), 23040260707644U);
        checked = true;
        break;
    }
    ASSERT_TRUE(checked) << "no cpu_thread track returned the expected 59 intervals";
}

TEST_F(reader_test, v3_get_track_stats_counter_matches_scalar_slice)
{
    auto tracks = m_reader->get_all_tracks();
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);

    auto samples = m_reader->get_scalar_track(counter->id);
    auto stats   = m_reader->get_track_stats(counter->id);
    expect_stats_match_scalars(stats, samples);
    ASSERT_GT(stats.count, 0U);
}

TEST_F(reader_test, v3_get_track_stats_unknown_id_returns_empty)
{
    // Unknown track id is not an error: zero count, nullopt bounds.
    constexpr size_t kUnknownTrackId = 999999999;
    auto             stats           = m_reader->get_track_stats(kUnknownTrackId);
    ASSERT_EQ(stats.count, 0U);
    ASSERT_FALSE(stats.min_ts.has_value());
    ASSERT_FALSE(stats.max_ts.has_value());
}

TEST_F(reader_test, v3_get_all_tracks_synthesizes_stream_track)
{
    // The capture has one stream (stream_id=0). Stream tracks aggregate three event
    // tables and are ADDITIVE to the gpu_queue/dma tracks (the same events also appear
    // there), so the sole stream is a distinct synthesized track keyed (nid,pid,0).
    auto tracks  = m_reader->get_all_tracks();
    auto streams = find_tracks(tracks, profiler_hub::reader_types::track_type_t::stream);
    ASSERT_EQ(streams.size(), 1U);

    const auto& s = streams.front();
    ASSERT_NE(s->stream_info, nullptr);
    ASSERT_EQ(s->stream_info->stream_id, 0U);
    ASSERT_NE(s->node_info, nullptr);
    ASSERT_EQ(s->node_info->node_id, 9162464413581981795);
    ASSERT_NE(s->process_info, nullptr);
    ASSERT_EQ(s->process_info->pid, 67979);
}

TEST_F(reader_test, v3_get_interval_track_stream_aggregates_ops_with_op_kind)
{
    // The stream track unions kernel_dispatch + memory_copy + memory_allocate that
    // share the stream. This capture's stream 0 has 1 dispatch + 2 copies + 0 allocs.
    // op_kind is retired: the event's opaque handle now encodes its type, so each
    // event's identity is proved by resolving it through exactly one detail accessor.
    auto tracks = m_reader->get_all_tracks();
    auto stream =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::stream);
    ASSERT_NE(stream, nullptr);

    auto intervals = m_reader->get_interval_track(stream->id);
    ASSERT_EQ(intervals.size(), 3U);
    ASSERT_TRUE(is_start_sorted(intervals));

    size_t kd = 0, mc = 0;
    for(const auto& ev : intervals)
    {
        ASSERT_GE(ev.end, ev.start);
        ASSERT_EQ(count_interval_resolutions(*m_reader, ev.id), 1)
            << "handle must resolve through exactly one detail accessor";
        if(m_reader->get_kernel_dispatch_details(ev.id).has_value())
            ++kd;
        else if(m_reader->get_memory_copy_details(ev.id).has_value())
            ++mc;
        else
            FAIL() << "unexpected event type on stream 0";
    }
    ASSERT_EQ(kd, 1U);
    ASSERT_EQ(mc, 2U);
}

TEST_F(reader_test, v3_get_track_stats_stream_matches_interval_slice)
{
    auto tracks = m_reader->get_all_tracks();
    auto stream =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::stream);
    ASSERT_NE(stream, nullptr);

    auto intervals = m_reader->get_interval_track(stream->id);
    auto stats     = m_reader->get_track_stats(stream->id);
    expect_stats_match_intervals(stats, intervals);
    ASSERT_EQ(stats.count, 3U);
}

// ============================================================================
// Track-scoped API tests — v3 synthetic edge-matrix fixture (rocpd_v3_edge.db)
// Built at configure time from fixtures/rocpd_v3_edge_data.sql + the canonical v3
// schema. Unlike the bundled real capture (rocpd.db), every row is hand-authored
// so tests assert KNOWN values and cover schema-permitted branches the real
// capture happens not to contain: a counter track WITH a tid, a cpu_thread track
// with NULL pid, multiple gpu_queue / dma lanes, and stack_id=0/NULL flow
// exclusion. See the fixture header for the full oracle.
// ============================================================================

class reader_v3_edge_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_V3_EDGE_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_edge_test, track_matrix_counts_by_type)
{
    // cpu_thread/region tracks are synthesized from rocpd_region, not rocpd_track.
    // rocpd_track contributes 4 PMC-backed sampled (counter) rows (2, 3, 6, 8);
    // the non-counter rows (1, 4, 5) are ignored, and track 7 -- sampled but with NO
    // rocpd_pmc_event -- is NOT a counter (see counter_discovery_excludes_non_pmc_sample
    // below). Track 8 (pmc_id 99, empty PMC name) IS a counter -- discovery joins
    // rocpd_pmc_event (present), not rocpd_info_pmc; it tests the display-name fallback.
    // Synthesis adds 1 cpu_thread, 2 gpu_queue, 1 dma, 2 stream, 1 memory => 11 tracks.
    // Task 012B adds 1 memory_activity (1 alloc row, agent_id=1) => total 12.
    auto tracks = m_reader->get_all_tracks();
    ASSERT_EQ(tracks.size(), 12U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::cpu_thread).size(),
        1U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter).size(),
        4U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::gpu_queue).size(),
        2U);
    ASSERT_EQ(find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma).size(),
              1U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::stream).size(), 2U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory).size(), 1U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory_activity)
            .size(),
        1U);
}

TEST_F(reader_v3_edge_test, counter_discovery_excludes_non_pmc_sample_track)
{
    // Regression: 005B-4-fix-4. Counter discovery must classify a track as a counter
    // only when a PMC-backed rocpd_sample references it (the sample's event_id joins
    // rocpd_pmc_event), NOT merely when any rocpd_sample references it. Track 7 in the
    // fixture has a rocpd_sample (sample 7 / event 14) but NO rocpd_pmc_event, so it is
    // a non-PMC sample track. The old "DISTINCT track_id FROM rocpd_sample" discovery
    // over-included such tracks as empty counters (the rocpd-transpose.db 21-vs-18
    // divergence); distinct_sample_track_ids() now joins rocpd_pmc_event, so track 7
    // must not appear as a counter -- and since it has no rocpd_region row, it must not
    // appear as any track type at all.
    auto tracks = m_reader->get_all_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);
    // Primary signal: only the 4 PMC-backed sample tracks (2, 3, 6, 8) are counters.
    // Track 7 is excluded (has no rocpd_pmc_event); track 8 is included (pmc_event
    // with pmc_id=99 -- the pmc row is absent from rocpd_info_pmc, but discovery
    // only needs the pmc_event join, not the pmc metadata row).
    ASSERT_EQ(counters.size(), 4U);
    // Corroborating signal: every counter is PMC-backed, so each resolves to a
    // non-empty scalar track. The spurious non-PMC track 7 would resolve to zero samples.
    for(const auto& c : counters)
        ASSERT_FALSE(m_reader->get_scalar_track(c->id).empty())
            << "counter track " << c->id << " has no PMC-backed samples";
}

TEST_F(reader_v3_edge_test, counter_identity_null_pid_and_null_tid_branches)
{
    // Re-homed from the former cpu_thread coverage: under region-synthesis, region
    // tracks always carry a real (nid,pid,tid), so the NULL-pid/NULL-tid identity
    // branches can no longer be exercised on cpu_thread tracks. v3 counter tracks
    // still come from rocpd_track (Q10) and CAN carry NULL pid/tid, so the same
    // nullable-identity matrix now lives here:
    //   track 2: pid set, tid NULL -> process_info set,  thread_info NULL
    //   track 3: pid + tid set     -> process_info set,  thread_info SET
    //   track 6: pid NULL          -> process_info NULL, thread_info NULL
    //   track 8: pid set, tid NULL -> process_info set,  thread_info NULL (fallback)
    auto tracks = m_reader->get_all_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_EQ(counters.size(), 4U);

    int with_thread = 0, with_process = 0, without_process = 0;
    for(const auto& t : counters)
    {
        if(t->thread_info != nullptr) ++with_thread;
        if(t->process_info != nullptr)
            ++with_process;
        else
            ++without_process;
    }
    // Exactly one counter track carries a resolved thread (tid set -- track 3).
    ASSERT_EQ(with_thread, 1);
    // Exactly one carries no process (pid NULL -- track 6); tracks 2/3/8 do.
    ASSERT_EQ(without_process, 1);
    ASSERT_EQ(with_process, 3);
}

TEST_F(reader_v3_edge_test, counter_thread_info_tracks_tid_agent_info_always_null)
{
    // The #147 contract, both branches. thread_info is driven by rocpd_track.tid
    // and is orthogonal to counter classification; agent_info is impossible on v3
    // (rocpd_track has no agent_id column) regardless of tid.
    auto tracks = m_reader->get_all_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_EQ(counters.size(), 4U);

    profiler_hub::reader_types::track_info_ptr_t no_tid_counter;    // GRBM_COUNT
    profiler_hub::reader_types::track_info_ptr_t with_tid_counter;  // SQ_WAVES
    for(const auto& c : counters)
    {
        if(c->name == "GRBM_COUNT")
            no_tid_counter = c;
        else if(c->name == "SQ_WAVES")
            with_tid_counter = c;
    }
    ASSERT_NE(no_tid_counter, nullptr)
        << "counter display name should be its PMC name (Q9)";
    ASSERT_NE(with_tid_counter, nullptr)
        << "counter display name should be its PMC name (Q9)";

    // Branch 1: counter with tid NULL -> thread_info null.
    ASSERT_EQ(no_tid_counter->thread_info, nullptr);
    ASSERT_EQ(no_tid_counter->agent_info, nullptr);

    // Branch 2: counter WITH tid -> thread_info populated (the case rocpd.db lacks).
    ASSERT_NE(with_tid_counter->thread_info, nullptr);
    ASSERT_EQ(with_tid_counter->agent_info, nullptr);
}

TEST_F(reader_v3_edge_test, counter_display_name_falls_back_to_track_name_on_pmc_miss)
{
    // F7 coverage: when the pmc_info lookup produces an empty name, the display name must
    // fall back to rocpd_track.name rather than being empty, zero-initialized, or stale.
    // Mechanism: reader_impl.cpp checks !nit->second.empty() before overwriting the name;
    // if the PMC name in rocpd_info_pmc is "" the guard fires and rocpd_track.name stays.
    // Track 8: rocpd_track.name_id=7 -> "FallbackCounter"; pmc_id=99 exists in
    // rocpd_info_pmc with an intentionally empty name field.
    auto tracks = m_reader->get_all_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);

    profiler_hub::reader_types::track_info_ptr_t fallback_counter;
    for(const auto& c : counters)
    {
        if(c->name == "FallbackCounter")
        {
            fallback_counter = c;
            break;
        }
    }
    ASSERT_NE(fallback_counter, nullptr) << "fallback counter track not found";

    // Primary assertion: display name equals rocpd_track.name (the fallback value).
    ASSERT_EQ(fallback_counter->name, "FallbackCounter");
    // Sanity: non-empty, not garbage.
    ASSERT_FALSE(fallback_counter->name.empty());
    // pmc_info: pmc_id=99 is in rocpd_info_pmc with empty name -> pmc_info IS attached
    // but carries an empty name, which is exactly what triggers the fallback guard.
    ASSERT_NE(fallback_counter->pmc_info, nullptr);
    ASSERT_TRUE(fallback_counter->pmc_info->name.empty());
    // Non-fallback path still intact: the 3 fully-resolved counters have non-empty names
    // and their display name equals the PMC name (name != track->name only for fallback).
    size_t with_pmc_name_match = 0;
    for(const auto& c : counters)
    {
        if(c->pmc_info != nullptr && !c->pmc_info->name.empty())
        {
            ASSERT_EQ(c->name, c->pmc_info->name);
            ++with_pmc_name_match;
        }
    }
    ASSERT_EQ(with_pmc_name_match, 3U);
}

TEST_F(reader_v3_edge_test, get_interval_track_cpu_thread_regions_ordered)
{
    // track 1 carries 4 regions; ORDER BY start (row-id order deliberately differs):
    //   region 2 (start 1000) -> 3 (2000) -> 1 (3000) -> 4 (6000).
    auto tracks = m_reader->get_all_tracks();
    auto cpu = find_tracks(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);

    profiler_hub::reader_types::interval_event_list_t regions;
    for(const auto& t : cpu)
    {
        auto iv = m_reader->get_interval_track(t->id);
        if(iv.size() == 4)
        {
            regions = std::move(iv);
            break;
        }
    }
    ASSERT_EQ(regions.size(), 4U);
    ASSERT_TRUE(is_start_sorted(regions));
    ASSERT_EQ(row_id_of(regions.front().id), 2U);
    ASSERT_EQ(regions.front().start, 1000);
    ASSERT_EQ(regions.front().end, 5000);

    auto details = m_reader->get_region_details(regions.front().id);
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->name, "RegionAlpha");
}

TEST_F(reader_v3_edge_test, get_interval_track_gpu_queue_and_dma_ordered)
{
    auto tracks = m_reader->get_all_tracks();

    // Two gpu_queue tracks: Queue-A has 2 dispatches (start 1200, 1600), Queue-B 1.
    auto gpu = find_tracks(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_EQ(gpu.size(), 2U);
    profiler_hub::reader_types::interval_event_list_t gpu_two;
    size_t                                            gpu_singletons = 0;
    for(const auto& t : gpu)
    {
        auto iv = m_reader->get_interval_track(t->id);
        if(iv.size() == 2)
            gpu_two = iv;
        else if(iv.size() == 1)
            ++gpu_singletons;
    }
    ASSERT_EQ(gpu_two.size(), 2U);
    ASSERT_EQ(gpu_singletons, 1U);
    ASSERT_TRUE(is_start_sorted(gpu_two));
    ASSERT_EQ(gpu_two.front().start, 1200);

    // One dma track (all 3 copies share queue_id NULL + dst_agent_id NULL under the
    // by-destination-agent key). Row-id order != start order proves ORDER BY start:
    // copies at 2200 (mc1), 2400 (mc2), 2100 (mc3) => [2100, 2200, 2400].
    auto dma = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    ASSERT_EQ(dma.size(), 1U);
    auto dma_iv = m_reader->get_interval_track(dma.front()->id);
    ASSERT_EQ(dma_iv.size(), 3U);
    ASSERT_TRUE(is_start_sorted(dma_iv));
    ASSERT_EQ(dma_iv.front().start, 2100);
}

TEST_F(reader_v3_edge_test, get_scalar_track_values_for_both_counters)
{
    auto tracks = m_reader->get_all_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);

    for(const auto& c : counters)
    {
        auto samples = m_reader->get_scalar_track(c->id);
        ASSERT_TRUE(is_timestamp_sorted(samples));

        if(c->name == "GRBM_COUNT")
        {
            // 3 samples, ascending timestamp despite differing row-id order.
            ASSERT_FALSE(samples.empty());
            ASSERT_EQ(samples.size(), 3U);
            ASSERT_EQ(samples.front().timestamp, 1000);
            ASSERT_DOUBLE_EQ(samples.front().value, 10.5);

            auto details = m_reader->get_scalar_details(samples.front().id);
            ASSERT_TRUE(details.has_value());
            ASSERT_DOUBLE_EQ(details->value, samples.front().value);
        }
        else if(c->name == "SQ_WAVES")
        {
            ASSERT_FALSE(samples.empty());
            ASSERT_EQ(samples.size(), 2U);
            ASSERT_EQ(samples.front().timestamp, 500);
            ASSERT_DOUBLE_EQ(samples.front().value, 5.0);

            auto details = m_reader->get_scalar_details(samples.front().id);
            ASSERT_TRUE(details.has_value());
            ASSERT_DOUBLE_EQ(details->value, samples.front().value);
        }
        // Track 8 ("FallbackCounter") has 1 sample with pmc_id=99 (empty name in
        // rocpd_info_pmc); no specific assertions here — coverage in
        // counter_display_name_falls_back_to_track_name_on_pmc_miss.
    }
}

TEST_F(reader_v3_edge_test, get_flows_excludes_zero_and_null_stack_id)
{
    // stack_id linkage (Q4): region<->kernel_dispatch (100), region<->memory_copy
    // (200), region<->memory_allocate (400) = 3 flows. RegionGamma (stack 0) and
    // the sample events (stack NULL) are excluded. Flat clique (one region + one
    // GPU event per stack) => region source, one GPU-type dest, no siblings.
    auto flows = m_reader->get_flows();
    ASSERT_EQ(flows.size(), 3U);
    for(const auto& f : flows)
    {
        ASSERT_GT(row_id_of(f.source), 0U);
        ASSERT_GT(row_id_of(f.dest), 0U);
        ASSERT_TRUE(m_reader->get_region_details(f.source).has_value());
        ASSERT_EQ(count_interval_resolutions(*m_reader, f.dest), 1);
        ASSERT_FALSE(m_reader->get_region_details(f.dest).has_value());
    }
}

TEST_F(reader_v3_edge_test, track_scoped_queries_respect_type)
{
    // Q7: interval query on a counter (scalar-only) track and scalar query on a
    // cpu_thread (interval-only) track both return empty, not an error.
    auto tracks = m_reader->get_all_tracks();
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(counter, nullptr);
    ASSERT_NE(cpu, nullptr);
    ASSERT_TRUE(m_reader->get_interval_track(counter->id).empty());
    ASSERT_TRUE(m_reader->get_scalar_track(cpu->id).empty());
}

TEST_F(reader_v3_edge_test, get_track_stats_matches_slices_for_every_track_type)
{
    // Hand-authored oracle: the 4-region cpu_thread track spans start 1000..end 6000+.
    // For every track, stats must equal MIN/MAX/COUNT over the exact interval/scalar
    // slice — this covers cpu_thread, gpu_queue, dma (here the "neither" variant:
    // queue_id NULL + dst_agent_id NULL; the queue+agent "qa" variant is covered by the
    // dma-by-agent fixture) and counter in one pass, per synthesized track flavor.
    auto tracks = m_reader->get_all_tracks();

    bool checked_cpu = false;
    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::cpu_thread))
    {
        auto intervals = m_reader->get_interval_track(t->id);
        auto stats     = m_reader->get_track_stats(t->id);
        expect_stats_match_intervals(stats, intervals);
        if(intervals.size() == 4)
        {
            ASSERT_EQ(stats.min_ts.value(), 1000U);
            checked_cpu = true;
        }
    }
    ASSERT_TRUE(checked_cpu) << "expected a 4-region cpu_thread track";

    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::gpu_queue))
    {
        auto intervals = m_reader->get_interval_track(t->id);
        expect_stats_match_intervals(m_reader->get_track_stats(t->id), intervals);
    }

    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma))
    {
        auto intervals = m_reader->get_interval_track(t->id);
        expect_stats_match_intervals(m_reader->get_track_stats(t->id), intervals);
    }

    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter))
    {
        auto samples = m_reader->get_scalar_track(t->id);
        expect_stats_match_scalars(m_reader->get_track_stats(t->id), samples);
    }

    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::stream))
    {
        auto intervals = m_reader->get_interval_track(t->id);
        expect_stats_match_intervals(m_reader->get_track_stats(t->id), intervals);
    }

    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory))
    {
        auto intervals = m_reader->get_interval_track(t->id);
        expect_stats_match_intervals(m_reader->get_track_stats(t->id), intervals);
    }

    for(const auto& t :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory_activity))
    {
        auto samples = m_reader->get_scalar_track(t->id);
        expect_stats_match_scalars(m_reader->get_track_stats(t->id), samples);
    }
}

TEST_F(reader_v3_edge_test, get_interval_track_stream_aggregates_three_op_kinds)
{
    // This is the only fixture exercising all THREE UNION legs of a stream track,
    // including memory_allocate (no real capture available to the project has a
    // memory_allocate row carrying a stream_id). Hand-authored oracle:
    //   stream 1 (nid,pid,stream_id)=(1,1,1): 3 kernel_dispatch + 2 memory_copy +
    //       1 memory_allocate = 6 events, ORDER BY start:
    //       kd3(1200) kd2(1400) kd1(1600) mc3(2100) mc1(2200) ma1(6100)
    //   stream 2 (1,1,2): 1 memory_copy = 1 event (mc2 start 2400)
    // op_kind is retired: each event's opaque handle encodes its type and resolves
    // through exactly one get_*_details() accessor, which is what we assert here.
    auto tracks  = m_reader->get_all_tracks();
    auto streams = find_tracks(tracks, profiler_hub::reader_types::track_type_t::stream);
    ASSERT_EQ(streams.size(), 2U);

    profiler_hub::reader_types::track_info_ptr_t s1, s2;
    for(const auto& s : streams)
    {
        ASSERT_NE(s->stream_info, nullptr);
        if(s->stream_info->stream_id == 1)
            s1 = s;
        else if(s->stream_info->stream_id == 2)
            s2 = s;
    }
    ASSERT_NE(s1, nullptr);
    ASSERT_NE(s2, nullptr);

    auto iv1 = m_reader->get_interval_track(s1->id);
    ASSERT_EQ(iv1.size(), 6U);
    ASSERT_TRUE(is_start_sorted(iv1));

    size_t kd = 0, mc = 0, ma = 0;
    for(const auto& ev : iv1)
    {
        ASSERT_EQ(count_interval_resolutions(*m_reader, ev.id), 1)
            << "handle must resolve through exactly one detail accessor";
        if(m_reader->get_kernel_dispatch_details(ev.id).has_value())
            ++kd;
        else if(m_reader->get_memory_copy_details(ev.id).has_value())
            ++mc;
        else if(m_reader->get_memory_alloc_details(ev.id).has_value())
            ++ma;
        else
            FAIL() << "unexpected event type on stream 1";
    }
    ASSERT_EQ(kd, 3U);
    ASSERT_EQ(mc, 2U);
    ASSERT_EQ(ma, 1U);
    ASSERT_EQ(iv1.front().start, 1200);
    ASSERT_TRUE(m_reader->get_kernel_dispatch_details(iv1.front().id).has_value());
    ASSERT_EQ(iv1.back().start, 6100);
    ASSERT_TRUE(m_reader->get_memory_alloc_details(iv1.back().id).has_value());

    auto iv2 = m_reader->get_interval_track(s2->id);
    ASSERT_EQ(iv2.size(), 1U);
    ASSERT_EQ(iv2.front().start, 2400);
    ASSERT_TRUE(m_reader->get_memory_copy_details(iv2.front().id).has_value());

    expect_stats_match_intervals(m_reader->get_track_stats(s1->id), iv1);
    expect_stats_match_intervals(m_reader->get_track_stats(s2->id), iv2);
}

TEST_F(reader_v3_edge_test, get_interval_track_stream_memalloc_event_carries_category)
{
    // 005B-2-fix-1 flagged gap: the memory_allocate UNION leg in the stream SQL carries
    // the category LEFT JOIN (same pattern as kd/mc legs) but no committed fixture
    // previously asserted a category value on a memalloc-in-stream event. The edge
    // fixture's sole memory_allocate row (ma1, event_id=7) has no category_id set, so
    // the resolved category must be an empty string — asserting that proves the
    // structural LEFT JOIN is executed correctly without silent breakage.
    auto tracks  = m_reader->get_all_tracks();
    auto streams = find_tracks(tracks, profiler_hub::reader_types::track_type_t::stream);

    profiler_hub::reader_types::track_info_ptr_t s1;
    for(const auto& s : streams)
    {
        if(s->stream_info && s->stream_info->stream_id == 1) s1 = s;
    }
    ASSERT_NE(s1, nullptr);

    auto iv = m_reader->get_interval_track(s1->id);
    ASSERT_EQ(iv.size(), 6U);

    bool found_ma = false;
    for(const auto& ev : iv)
    {
        // The memory_allocate leg is identified by resolving the handle through the
        // memory-alloc detail accessor (op_kind is retired).
        if(m_reader->get_memory_alloc_details(ev.id).has_value())
        {
            // event_id=7 has no category_id in the edge fixture -> LEFT JOIN yields NULL
            // -> category resolves to empty string (not a missing field, not a crash).
            EXPECT_EQ(ev.category, "");
            found_ma = true;
        }
    }
    EXPECT_TRUE(found_ma) << "stream 1 must contain at least one memory_allocate event";
}

TEST_F(reader_v3_edge_test, get_interval_track_memory_type_interval_and_identity)
{
    // task 009 added track_type_t::memory for rocpd_memory_allocate rows keyed by
    // (nid, agent_id, queue_id, pid). The edge fixture has one such row:
    //   (id=1, nid=1, pid=1, agent_id=1, type='ALLOC', start=6100, end=6200, size=4096,
    //    queue_id=NULL, stream_id=1, event_id=7).
    // This exercises the "a_only" variant (agent_id set, queue_id NULL).
    // No test previously called get_interval_track() on a memory track; this is the
    // gap identified by the task-007 audit.
    auto tracks  = m_reader->get_all_tracks();
    auto mem_trk = find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory);
    ASSERT_EQ(mem_trk.size(), 1U);

    const auto& t = mem_trk.front();
    // agent_info must be populated (agent_id=1); queue_info null (queue_id IS NULL).
    ASSERT_NE(t->agent_info, nullptr);
    EXPECT_EQ(t->agent_info->id, 1U);
    EXPECT_EQ(t->queue_info, nullptr);

    auto intervals = m_reader->get_interval_track(t->id);
    ASSERT_EQ(intervals.size(), 1U);
    EXPECT_EQ(intervals.front().start, 6100U);
    EXPECT_EQ(intervals.front().end, 6200U);

    // the handle must resolve through get_memory_alloc_details().
    auto details = m_reader->get_memory_alloc_details(intervals.front().id);
    ASSERT_TRUE(details.has_value());
    EXPECT_EQ(details->start_timestamp, 6100U);
    EXPECT_EQ(details->end_timestamp, 6200U);
    EXPECT_EQ(details->size, 4096U);
    EXPECT_EQ(details->type, "ALLOC");
}

TEST_F(reader_v3_edge_test, get_track_stats_memory_type_matches_interval_slice)
{
    // get_track_stats() must return the same count/min/max as the interval slice for
    // the memory track — not previously covered (gap from task-007 audit).
    auto tracks  = m_reader->get_all_tracks();
    auto mem_trk = find_tracks(tracks, profiler_hub::reader_types::track_type_t::memory);
    ASSERT_EQ(mem_trk.size(), 1U);

    auto intervals = m_reader->get_interval_track(mem_trk.front()->id);
    expect_stats_match_intervals(m_reader->get_track_stats(mem_trk.front()->id),
                                 intervals);
}

// ============================================================================
// get_flows() full-clique tests — v3 synthetic clique fixture (rocpd_v3_clique.db)
// Built at configure time from fixtures/rocpd_v3_clique_data.sql + the canonical
// v3 schema. The edge fixture above is a FLAT clique (one region + one GPU event
// per stack) so it proves neither the new region->region / same-type sibling
// categories nor the endpoint-id collision. This fixture authors non-flat stack
// cliques whose endpoint ids deliberately collide across type tables, so the
// event_type tags are the ONLY disambiguator. See the fixture header for the
// full by-construction oracle (11 flows: rkd=1 rmc=1 rma=1 rr=2 kdkd=2 mcmc=2
// mama=2).
// ============================================================================

class reader_v3_clique_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_V3_CLIQUE_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_clique_test, get_flows_emits_full_clique_with_typed_endpoints)
{
    using et         = profiler_hub::reader_types::event_type_t;
    using flow_key_t = std::pair<profiler_hub::reader_types::event_id_t,
                                 profiler_hub::reader_types::event_id_t>;

    auto flows = m_reader->get_flows();
    ASSERT_EQ(flows.size(), 11U);

    std::multiset<flow_key_t> got;
    for(const auto& f : flows)
    {
        got.emplace(f.source, f.dest);
    }

    // Exact by-construction oracle (see fixture header). The colliding endpoint ids
    // (region 1 / kd 1 / mc 1 / ma 1) share raw per-type row ids across tables, so the
    // minted opaque handles are equal ONLY when both the encoded type AND row id match.
    // This is the whole point of the fix: the endpoints are distinguishable without any
    // companion type tag.
    const std::multiset<flow_key_t> expected{
        { make_event_id(et::region, 1), make_event_id(et::kernel_dispatch, 1) },  // s1000
        { make_event_id(et::region, 1), make_event_id(et::memory_copy, 1) },      // s1000
        { make_event_id(et::region, 1), make_event_id(et::memory_allocate, 1) },  // s1000
        { make_event_id(et::region, 2), make_event_id(et::region, 3) },           // s2000
        { make_event_id(et::region, 3), make_event_id(et::region, 2) },           // s2000
        { make_event_id(et::kernel_dispatch, 2),
          make_event_id(et::kernel_dispatch, 3) },  // stack 3000
        { make_event_id(et::kernel_dispatch, 3),
          make_event_id(et::kernel_dispatch, 2) },                                 // 3000
        { make_event_id(et::memory_copy, 2), make_event_id(et::memory_copy, 3) },  // 4000
        { make_event_id(et::memory_copy, 3), make_event_id(et::memory_copy, 2) },  // 4000
        { make_event_id(et::memory_allocate, 2), make_event_id(et::memory_allocate, 3) },
        { make_event_id(et::memory_allocate, 3), make_event_id(et::memory_allocate, 2) },
    };
    ASSERT_EQ(got, expected);

    // Handle-collision guard: region 1 / kernel_dispatch 1 / memory_copy 1 /
    // memory_allocate 1 all share raw row id 1 but come from different per-type tables.
    // They MUST mint to four distinct handles (the identity leak this task closes).
    std::unordered_set<profiler_hub::reader_types::event_id_t> distinct{
        make_event_id(et::region, 1),
        make_event_id(et::kernel_dispatch, 1),
        make_event_id(et::memory_copy, 1),
        make_event_id(et::memory_allocate, 1)
    };
    ASSERT_EQ(distinct.size(), 4U);
}

// ============================================================================
// Track-scoped API tests — v4.0 real fixture (rocpd_v4.db)
// cpu_thread + gpu_queue + dma interval tracks and flows. This fixture has no
// counter samples, so the scalar path is covered by reader_v4_counter_test.
// ============================================================================

class reader_v4_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_V4_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v4_test, v4_track_classification_and_identity)
{
    auto tracks = m_reader->get_all_tracks();
    // Fixture has 5 tracks: 1 cpu_thread, 1 gpu_queue, 2 dma, 1 stream (the sole
    // rocpd_track.stream_id=0, aggregating kernel_dispatch + memory_copy).
    ASSERT_EQ(tracks.size(), 5);

    auto cpu = find_tracks(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    auto gpu = find_tracks(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    auto dma = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    ASSERT_EQ(cpu.size(), 1);
    ASSERT_EQ(gpu.size(), 1);
    ASSERT_EQ(dma.size(), 2);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::stream).size(), 1U);

    // gpu_queue carries agent + queue identity (Q10: v4 GPU tracks scoped to agent).
    ASSERT_NE(gpu[0]->agent_info, nullptr);
    ASSERT_EQ(gpu[0]->agent_info->name, "AMD Instinct MI300X");
    ASSERT_NE(gpu[0]->queue_info, nullptr);
    ASSERT_EQ(gpu[0]->queue_info->name, "Queue 0");

    // dma tracks carry agent identity; the fixture has one GPU-side and one CPU-side.
    bool saw_gpu_agent = false, saw_cpu_agent = false;
    for(const auto& d : dma)
    {
        ASSERT_NE(d->agent_info, nullptr);
        if(d->agent_info->agent_type == "GPU") saw_gpu_agent = true;
        if(d->agent_info->agent_type == "CPU") saw_cpu_agent = true;
    }
    ASSERT_TRUE(saw_gpu_agent);
    ASSERT_TRUE(saw_cpu_agent);
}

TEST_F(reader_v4_test, v4_get_interval_track_cpu_thread_regions)
{
    auto tracks = m_reader->get_all_tracks();
    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);

    auto intervals = m_reader->get_interval_track(cpu->id);
    ASSERT_EQ(intervals.size(), 384);
    ASSERT_TRUE(is_start_sorted(intervals));

    const auto& first = intervals.front();
    ASSERT_EQ(first.start, 516609802359041);
    ASSERT_EQ(first.end, 516609802359341);
    ASSERT_GE(first.end, first.start);
    ASSERT_GT(row_id_of(first.id), 0U);

    // The handle resolves through the region detail path.
    ASSERT_TRUE(m_reader->get_region_details(first.id).has_value());
}

TEST_F(reader_v4_test, v4_get_interval_track_cpu_thread_carries_category)
{
    // v4 resolves category through rocpd_info_category (a different table than v3's
    // rocpd_string), so this exercises the v4 branch of the per-backend resolution.
    // All 384 regions in this capture are "hsa_api"; assert the carried category
    // matches the detail-path oracle for every interval.
    auto tracks = m_reader->get_all_tracks();
    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);

    auto intervals = m_reader->get_interval_track(cpu->id);
    ASSERT_EQ(intervals.size(), 384);

    ASSERT_EQ(intervals.front().category, "hsa_api");
    for(const auto& ev : intervals)
    {
        auto details = m_reader->get_region_details(ev.id);
        ASSERT_TRUE(details.has_value());
        ASSERT_NE(details->event, nullptr);
        ASSERT_EQ(ev.category, details->event->event_category);
        ASSERT_EQ(ev.category, "hsa_api");
    }
}

TEST_F(reader_v4_test, v4_get_interval_track_gpu_queue_dispatches)
{
    auto tracks = m_reader->get_all_tracks();
    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);

    auto intervals = m_reader->get_interval_track(gpu->id);
    ASSERT_EQ(intervals.size(), 20);
    ASSERT_TRUE(is_start_sorted(intervals));
    ASSERT_EQ(intervals.front().start, 516609921772013);
    ASSERT_EQ(intervals.front().end, 516609921781427);

    // The handle resolves through the kernel dispatch detail path.
    ASSERT_TRUE(m_reader->get_kernel_dispatch_details(intervals.front().id).has_value());
}

TEST_F(reader_v4_test, v4_gpu_queue_track_carries_agent_id)
{
    // Same raw-agent-id contract as v3, exercised on the v4 backend (agent_id lives
    // on rocpd_track here). Fixture: the sole gpu_queue belongs to agent_id=6.
    auto tracks = m_reader->get_all_tracks();
    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);
    ASSERT_NE(gpu->agent_info, nullptr);
    ASSERT_EQ(gpu->agent_info->id, 6);
    ASSERT_EQ(gpu->agent_info->agent_type, "GPU");
}

TEST_F(reader_v4_test, v4_get_interval_track_gpu_queue_carries_category)
{
    // v4 resolves gpu_queue kernel-dispatch category through rocpd_info_category (a
    // different table than v3's rocpd_string), exercising the v4 branch. All 20
    // dispatches are "kernel_dispatch"; assert each carried category matches the
    // detail-path oracle.
    auto tracks = m_reader->get_all_tracks();
    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);

    auto intervals = m_reader->get_interval_track(gpu->id);
    ASSERT_EQ(intervals.size(), 20U);

    ASSERT_EQ(intervals.front().category, "kernel_dispatch");
    for(const auto& ev : intervals)
    {
        auto details = m_reader->get_kernel_dispatch_details(ev.id);
        ASSERT_TRUE(details.has_value());
        ASSERT_NE(details->event, nullptr);
        ASSERT_EQ(ev.category, details->event->event_category);
        ASSERT_EQ(ev.category, "kernel_dispatch");
    }
}

TEST_F(reader_v4_test, v4_get_interval_track_dma_memory_copies)
{
    auto tracks = m_reader->get_all_tracks();
    auto dma    = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    ASSERT_EQ(dma.size(), 2);

    for(const auto& d : dma)
    {
        auto intervals = m_reader->get_interval_track(d->id);
        ASSERT_EQ(intervals.size(), 1);
        ASSERT_GE(intervals.front().end, intervals.front().start);
        ASSERT_TRUE(m_reader->get_memory_copy_details(intervals.front().id).has_value());
    }
}

TEST_F(reader_v4_test, v4_get_interval_track_dma_carries_category)
{
    // v4 resolves memory-copy category through rocpd_info_category (a different table
    // than v3's rocpd_string), exercising the v4 branch. Both dma tracks hold a single
    // "memory_copy" interval; assert each carried category matches the detail-path
    // oracle.
    auto tracks = m_reader->get_all_tracks();
    auto dma    = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    ASSERT_EQ(dma.size(), 2);

    for(const auto& d : dma)
    {
        auto intervals = m_reader->get_interval_track(d->id);
        ASSERT_EQ(intervals.size(), 1U);
        for(const auto& ev : intervals)
        {
            ASSERT_EQ(ev.category, "memory_copy");
            auto details = m_reader->get_memory_copy_details(ev.id);
            ASSERT_TRUE(details.has_value());
            ASSERT_NE(details->event, nullptr);
            ASSERT_EQ(ev.category, details->event->event_category);
        }
    }
}

TEST_F(reader_v4_test, v4_get_scalar_track_on_interval_track_returns_empty)
{
    // Q7: scalar query against a gpu_queue (non-counter) track returns empty.
    auto tracks = m_reader->get_all_tracks();
    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);
    ASSERT_TRUE(m_reader->get_scalar_track(gpu->id).empty());
}

TEST_F(reader_v4_test, v4_get_flows_links_regions_to_gpu_events)
{
    // v4 fixture flows: 20 region->kernel_dispatch + 2 region->memory_copy = 22.
    // Flat clique, so the new categories add nothing; this asserts type-tag parity
    // with the v3 backend (every source is a region; dest is a GPU-side type).
    auto flows = m_reader->get_flows();
    ASSERT_EQ(flows.size(), 22);
    for(const auto& f : flows)
    {
        ASSERT_GT(row_id_of(f.source), 0U);
        ASSERT_GT(row_id_of(f.dest), 0U);
        ASSERT_TRUE(m_reader->get_region_details(f.source).has_value());
        ASSERT_EQ(count_interval_resolutions(*m_reader, f.dest), 1);
        ASSERT_FALSE(m_reader->get_region_details(f.dest).has_value());
    }
}

TEST_F(reader_v4_test, v4_get_track_stats_matches_slices_for_interval_tracks)
{
    // v4.0 tracks are canonical rocpd_track rows: stats resolve MIN/MAX through the
    // timestamp spine (start_id/end_id -> rocpd_timestamp). Cross-check every
    // interval track against its slice, plus pin the known cpu_thread bounds.
    auto tracks = m_reader->get_all_tracks();

    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);
    auto cpu_intervals = m_reader->get_interval_track(cpu->id);
    auto cpu_stats     = m_reader->get_track_stats(cpu->id);
    expect_stats_match_intervals(cpu_stats, cpu_intervals);
    ASSERT_EQ(cpu_stats.count, 384U);
    ASSERT_EQ(cpu_stats.min_ts.value(), 516609802359041U);

    auto gpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    ASSERT_NE(gpu, nullptr);
    auto gpu_intervals = m_reader->get_interval_track(gpu->id);
    auto gpu_stats     = m_reader->get_track_stats(gpu->id);
    expect_stats_match_intervals(gpu_stats, gpu_intervals);
    ASSERT_EQ(gpu_stats.count, 20U);
    ASSERT_EQ(gpu_stats.min_ts.value(), 516609921772013U);

    for(const auto& d :
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma))
    {
        auto intervals = m_reader->get_interval_track(d->id);
        expect_stats_match_intervals(m_reader->get_track_stats(d->id), intervals);
    }
}

TEST_F(reader_v4_test, v4_get_interval_track_stream_aggregates_ops_with_op_kind)
{
    // v4 stream tracks are synthesized from DISTINCT (nid,pid,stream_id) on
    // rocpd_track; each UNION leg JOINs rocpd_track ON stream_id and resolves times
    // through the timestamp spine. This capture's sole stream (stream_id=0) unions
    // 20 kernel_dispatch + 2 memory_copy + 0 memory_allocate = 22 events. The stream
    // aggregates ACROSS ops, so its earliest start (a memory_copy at 516609915990946)
    // precedes the gpu_queue's first dispatch (516609921772013) — proof the stream is
    // not just the queue track relabeled. The event's opaque handle encodes its type,
    // so the matching get_*_details() resolves for every event (op_kind is retired).
    auto tracks = m_reader->get_all_tracks();
    auto stream =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::stream);
    ASSERT_NE(stream, nullptr);

    auto intervals = m_reader->get_interval_track(stream->id);
    ASSERT_EQ(intervals.size(), 22U);
    ASSERT_TRUE(is_start_sorted(intervals));
    ASSERT_EQ(intervals.front().start, 516609915990946);

    size_t kd = 0, mc = 0;
    for(const auto& ev : intervals)
    {
        ASSERT_GE(ev.end, ev.start);
        ASSERT_EQ(count_interval_resolutions(*m_reader, ev.id), 1)
            << "handle must resolve through exactly one detail accessor";
        if(m_reader->get_kernel_dispatch_details(ev.id).has_value())
            ++kd;
        else if(m_reader->get_memory_copy_details(ev.id).has_value())
            ++mc;
        else
            FAIL() << "unexpected event type on stream 0";
    }
    ASSERT_EQ(kd, 20U);
    ASSERT_EQ(mc, 2U);
}

TEST_F(reader_v4_test, v4_get_track_stats_stream_matches_interval_slice)
{
    auto tracks = m_reader->get_all_tracks();
    auto stream =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::stream);
    ASSERT_NE(stream, nullptr);

    auto intervals = m_reader->get_interval_track(stream->id);
    auto stats     = m_reader->get_track_stats(stream->id);
    expect_stats_match_intervals(stats, intervals);
    ASSERT_EQ(stats.count, 22U);
}

// ============================================================================
// Track-scoped API tests — v4.0 synthetic counter fixture (rocpd_v4_counter.db)
// Built at configure time from committed SQL. Exists solely to exercise the
// v4.0 scalar/counter path (get_scalar_track / get_scalar_details), which no
// real v4.0 capture available to the project contains (no rocpd_sample rows).
// ============================================================================

class reader_v4_counter_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_V4_COUNTER_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v4_counter_test, v4_counter_track_classified_named_and_agent_scoped)
{
    auto tracks = m_reader->get_all_tracks();
    // Two tracks: the counter track (sample-referenced) and a bare cpu_thread.
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);

    // Q9: counter track display name is the PMC name.
    ASSERT_EQ(counter->name, "GRBM_COUNT");
    // Q10: v4 counter track carries agent_info (its rocpd_track row has agent_id).
    ASSERT_NE(counter->agent_info, nullptr);
    ASSERT_NE(counter->thread_info, nullptr);

    // v4.0 has one pmc per event (no event_id fan-out), so it is unaffected by the
    // v3-only deterministic disambiguation (005B-4-fix-1-fix-1): the single track must
    // still resolve to the GRBM_COUNT pmc, with name/agent consistent with the track.
    ASSERT_NE(counter->pmc_info, nullptr);
    ASSERT_EQ(counter->pmc_info->name, "GRBM_COUNT");
    // 005B-4-fix-1-fix-2: numeric pmc_id exposed on pmc_info; GRBM_COUNT is pmc 1 here.
    ASSERT_EQ(counter->pmc_info->pmc_id, 1U);
    ASSERT_EQ(counter->name, counter->pmc_info->name);
    ASSERT_NE(counter->pmc_info->agent_info, nullptr);
    ASSERT_EQ(counter->pmc_info->agent_info->agent_type, "GPU");
    ASSERT_EQ(counter->pmc_info->agent_info->type_index, 0U);
}

TEST_F(reader_v4_counter_test, v4_get_scalar_track_returns_timestamp_ordered_values)
{
    auto tracks = m_reader->get_all_tracks();
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);

    auto samples = m_reader->get_scalar_track(counter->id);
    // 3 samples, returned in ascending-timestamp order despite row-id order differing.
    ASSERT_EQ(samples.size(), 3);
    ASSERT_TRUE(is_timestamp_sorted(samples));

    ASSERT_EQ(row_id_of(samples[0].id), 2U);
    ASSERT_EQ(samples[0].timestamp, 1000);
    ASSERT_DOUBLE_EQ(samples[0].value, 10.5);

    ASSERT_EQ(row_id_of(samples[1].id), 3U);
    ASSERT_EQ(samples[1].timestamp, 2000);
    ASSERT_DOUBLE_EQ(samples[1].value, 20.5);

    ASSERT_EQ(row_id_of(samples[2].id), 1U);
    ASSERT_EQ(samples[2].timestamp, 3000);
    ASSERT_DOUBLE_EQ(samples[2].value, 30.5);
}

TEST_F(reader_v4_counter_test, v4_get_scalar_details_resolves_value_and_timestamp)
{
    // sample row id 1 -> timestamp 3000, value 30.5. The scalar handle encodes the
    // sample event type, which get_scalar_details() requires.
    auto details = m_reader->get_scalar_details(
        make_event_id(profiler_hub::reader_types::event_type_t::sample, 1));
    ASSERT_TRUE(details.has_value());
    ASSERT_DOUBLE_EQ(details->value, 30.5);
    ASSERT_EQ(details->sample.timestamp, 3000);
    ASSERT_NE(details->sample.track, nullptr);
}

TEST_F(reader_v4_counter_test, v4_get_interval_track_on_counter_returns_empty)
{
    // Q7: interval query against the counter track returns empty.
    auto tracks = m_reader->get_all_tracks();
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);
    ASSERT_TRUE(m_reader->get_interval_track(counter->id).empty());
}

TEST_F(reader_v4_counter_test, v4_get_scalar_track_on_non_counter_returns_empty)
{
    // Q7: scalar query against the bare cpu_thread track (no samples) returns empty.
    auto tracks = m_reader->get_all_tracks();
    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);
    ASSERT_TRUE(m_reader->get_scalar_track(cpu->id).empty());
}

TEST_F(reader_v4_counter_test, v4_get_track_stats_counter_matches_scalar_slice)
{
    // v4.0 scalar stats resolve MIN/MAX through the timestamp spine. Known oracle:
    // 3 samples at timestamps 1000/2000/3000 -> min 1000, max 3000, count 3.
    auto tracks = m_reader->get_all_tracks();
    auto counter =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(counter, nullptr);

    auto samples = m_reader->get_scalar_track(counter->id);
    auto stats   = m_reader->get_track_stats(counter->id);
    expect_stats_match_scalars(stats, samples);
    ASSERT_EQ(stats.count, 3U);
    ASSERT_EQ(stats.min_ts.value(), 1000U);
    ASSERT_EQ(stats.max_ts.value(), 3000U);
}

TEST_F(reader_v4_counter_test, v4_get_track_stats_bare_cpu_thread_is_empty)
{
    // The bare cpu_thread track has no region rows: count 0, nullopt bounds — the
    // honest "empty track" signal (SQL MIN/MAX over an empty set), not an error.
    auto tracks = m_reader->get_all_tracks();
    auto cpu =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    ASSERT_NE(cpu, nullptr);

    auto stats = m_reader->get_track_stats(cpu->id);
    ASSERT_EQ(stats.count, 0U);
    ASSERT_FALSE(stats.min_ts.has_value());
    ASSERT_FALSE(stats.max_ts.has_value());
}

TEST_F(reader_v4_counter_test,
       v4_counter_display_name_falls_back_to_track_name_on_pmc_miss)
{
    // F7 coverage (v4 backend): track 3 has rocpd_track.name_id=2 -> 'FallbackCounterV4';
    // its pmc_event references pmc_id=99 which exists in rocpd_info_pmc with empty name.
    // The empty-name guard (!nit->second.empty()) prevents it from overwriting
    // rocpd_track.name -> display name falls back to "FallbackCounterV4".
    auto tracks = m_reader->get_all_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);

    profiler_hub::reader_types::track_info_ptr_t fallback_counter;
    for(const auto& c : counters)
    {
        if(c->name == "FallbackCounterV4")
        {
            fallback_counter = c;
            break;
        }
    }
    ASSERT_NE(fallback_counter, nullptr) << "v4 fallback counter track not found";

    // Primary assertion: display name equals rocpd_track.name (the fallback value).
    ASSERT_EQ(fallback_counter->name, "FallbackCounterV4");
    ASSERT_FALSE(fallback_counter->name.empty());
    // pmc_info is attached (pmc_id=99 exists in rocpd_info_pmc) but carries empty name.
    ASSERT_NE(fallback_counter->pmc_info, nullptr);
    ASSERT_TRUE(fallback_counter->pmc_info->name.empty());
    // Non-fallback path still intact: the GRBM_COUNT track carries pmc_info.
    auto grbm =
        find_first_track(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_NE(grbm, nullptr);
    ASSERT_EQ(grbm->name, "GRBM_COUNT");
    ASSERT_NE(grbm->pmc_info, nullptr);
}

// v3 dma-by-destination-agent fixture: the crossed 2-agent x 2-stream x 12 = 48
// memory_copy pattern (fixtures/rocpd_v3_dma_agent_data.sql) that proves dma tracks
// partition by dst_agent_id, not stream_id -- the reproducible in-tree stand-in for
// roc-optiq's rocpd-transpose.db.
class reader_v3_dma_agent_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string m_database_path{ ROCPD_DB_V3_DMA_AGENT_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_dma_agent_test, dma_tracks_partition_by_destination_agent)
{
    // The 48 memory copies fully cross two destination agents (id 1, 2) with two
    // streams (12 events per agent/stream cell), all on one queue. Keyed by
    // (nid,pid,queue_id,dst_agent_id) this MUST yield exactly 2 dma tracks -- one per
    // destination agent, 24 events each -- matching Optiq's
    // GetRocprofMemoryCopyTrackQuery by-agent swimlane grouping. The old stream-keyed
    // identity would instead have given 2 tracks of 24 split BY STREAM, each spanning
    // both agents: the exact inverse. This test pins the by-agent partition and guards
    // against a regression back to by-stream.
    auto tracks = m_reader->get_all_tracks();
    auto dma    = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    ASSERT_EQ(dma.size(), 2U);

    std::set<size_t> track_agent_ids;
    for(const auto& track : dma)
    {
        // Every dma track carries agent_info resolved from its dst_agent_id (stream_info
        // is left null on dma tracks under the by-agent key).
        ASSERT_NE(track->agent_info, nullptr)
            << "dma track must resolve agent_info from dst_agent_id";
        ASSERT_EQ(track->stream_info, nullptr)
            << "dma track must not carry stream_info under the by-agent key";
        track_agent_ids.insert(track->agent_info->id);

        auto intervals = m_reader->get_interval_track(track->id);
        ASSERT_EQ(intervals.size(), 24U)
            << "each destination-agent track holds 24 copies (12 per stream)";

        // Membership proof: every copy in this track targets the SAME destination agent
        // as the track, and the track's copies span BOTH streams (proving the partition
        // is by agent, not by stream). copyStreamX/copyStreamY name each copy's stream.
        std::set<std::string> stream_names;
        for(const auto& ev : intervals)
        {
            auto details = m_reader->get_memory_copy_details(ev.id);
            ASSERT_TRUE(details.has_value());
            ASSERT_NE(details->dst_agent_id, nullptr);
            ASSERT_EQ(details->dst_agent_id->id, track->agent_info->id)
                << "copy in a dma track must target that track's destination agent";
            stream_names.insert(ev.display_name);
        }
        ASSERT_EQ(stream_names.size(), 2U) << "a destination-agent track must span both "
                                              "streams (by-agent, not by-stream)";
        ASSERT_TRUE(stream_names.count("copyStreamX") == 1);
        ASSERT_TRUE(stream_names.count("copyStreamY") == 1);
    }

    // The two tracks resolve to the two distinct destination agents (id 1 and 2).
    ASSERT_EQ(track_agent_ids.size(), 2U);
    ASSERT_TRUE(track_agent_ids.count(1) == 1);
    ASSERT_TRUE(track_agent_ids.count(2) == 1);
}

// ============================================================================
// kernel_dispatch_pmc track type — v3 synthetic fixture (rocpd_v3_kd_pmc.db)
// Data: 1 agent, 2 PMC types (SQ_WAVES pmc_id=1, GRBM_COUNT pmc_id=2),
// 3 dispatches: kd 1+2 on SQ_WAVES (start 1000,2000), kd 3 on GRBM_COUNT
// (start 3000). Tracks: (nid=1,agent_id=1,pmc_id=1,pid=100) has 2 events;
// (nid=1,agent_id=1,pmc_id=2,pid=100) has 1 event.
// ============================================================================

class reader_v3_kd_pmc_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_V3_KD_PMC_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_kd_pmc_test, v3_discovers_two_kd_pmc_tracks)
{
    // Two distinct (nid, agent_id, pmc_id, pid) -> 2 kernel_dispatch_pmc tracks.
    auto tracks =
        find_tracks(m_reader->get_all_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_EQ(tracks.size(), 2U);

    // Every kd_pmc track must carry agent_info (from agent_id=1).
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->agent_info, nullptr);
        ASSERT_EQ(t->agent_info->id, 1U);
        ASSERT_NE(t->process_info, nullptr);
        ASSERT_EQ(t->process_info->pid, 100U);
        ASSERT_NE(t->node_info, nullptr);
    }
}

TEST_F(reader_v3_kd_pmc_test, v3_kd_pmc_pmc_info_populated)
{
    // pmc_info must be resolved from pmc_id for both tracks.
    auto tracks =
        find_tracks(m_reader->get_all_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_EQ(tracks.size(), 2U);

    std::set<std::string> pmc_names;
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->pmc_info, nullptr);
        pmc_names.insert(t->pmc_info->name);
    }
    ASSERT_TRUE(pmc_names.count("SQ_WAVES") == 1);
    ASSERT_TRUE(pmc_names.count("GRBM_COUNT") == 1);
}

TEST_F(reader_v3_kd_pmc_test, v3_kd_pmc_interval_track_count_and_order)
{
    // The SQ_WAVES track (pmc_id=1) covers kd 1 (start=1000) and kd 2 (start=2000).
    // Rows are inserted out of start order (kd 2 first), so this proves ORDER BY start.
    auto tracks =
        find_tracks(m_reader->get_all_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_EQ(tracks.size(), 2U);

    profiler_hub::reader_types::track_info_ptr_t sq_waves_track;
    profiler_hub::reader_types::track_info_ptr_t grbm_track;
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->pmc_info, nullptr);
        if(t->pmc_info->name == "SQ_WAVES")
            sq_waves_track = t;
        else if(t->pmc_info->name == "GRBM_COUNT")
            grbm_track = t;
    }
    ASSERT_NE(sq_waves_track, nullptr);
    ASSERT_NE(grbm_track, nullptr);

    // SQ_WAVES track: 2 events in ascending start order.
    auto sq_intervals = m_reader->get_interval_track(sq_waves_track->id);
    ASSERT_EQ(sq_intervals.size(), 2U);
    ASSERT_TRUE(is_start_sorted(sq_intervals));
    ASSERT_EQ(sq_intervals[0].start, 1000U);
    ASSERT_EQ(sq_intervals[0].end, 1200U);
    ASSERT_EQ(sq_intervals[1].start, 2000U);
    ASSERT_EQ(sq_intervals[1].end, 2300U);

    // GRBM_COUNT track: 1 event.
    auto grbm_intervals = m_reader->get_interval_track(grbm_track->id);
    ASSERT_EQ(grbm_intervals.size(), 1U);
    ASSERT_EQ(grbm_intervals[0].start, 3000U);
    ASSERT_EQ(grbm_intervals[0].end, 3100U);
}

TEST_F(reader_v3_kd_pmc_test, v3_kd_pmc_track_stats_matches_interval_slice)
{
    auto tracks =
        find_tracks(m_reader->get_all_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    for(const auto& t : tracks)
    {
        auto intervals = m_reader->get_interval_track(t->id);
        auto stats     = m_reader->get_track_stats(t->id);
        expect_stats_match_intervals(stats, intervals);
    }
}

TEST_F(reader_v3_kd_pmc_test, v3_kd_pmc_display_name_from_kernel_symbol)
{
    // Interval display_name must be resolved from kernel_symbol (vecAdd(float*, int)).
    auto tracks =
        find_tracks(m_reader->get_all_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_GE(tracks.size(), 1U);
    auto intervals = m_reader->get_interval_track(tracks.front()->id);
    ASSERT_FALSE(intervals.empty());
    for(const auto& ev : intervals)
    {
        ASSERT_EQ(ev.display_name, "vecAdd(float*, int)");
    }
}

TEST_F(reader_v3_kd_pmc_test, v3_get_scalar_track_returns_empty_for_kd_pmc)
{
    // kernel_dispatch_pmc is an interval track; scalar read must return empty (Q7 guard).
    auto tracks =
        find_tracks(m_reader->get_all_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_GE(tracks.size(), 1U);
    ASSERT_TRUE(m_reader->get_scalar_track(tracks.front()->id).empty());
}

// ============================================================================
// kernel_dispatch_pmc track type — v4 synthetic fixture (rocpd_v4_kd_pmc.db)
// Mirrors the v3 fixture data shape; the presence of rocpd_timestamp triggers
// the v4 backend. Verifies the 4-arg timestamp-spine SQL path.
// ============================================================================

class reader_v4_kd_pmc_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_V4_KD_PMC_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v4_kd_pmc_test, v4_discovers_two_kd_pmc_tracks)
{
    auto tracks =
        find_tracks(m_reader->get_all_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_EQ(tracks.size(), 2U);

    for(const auto& t : tracks)
    {
        ASSERT_NE(t->agent_info, nullptr);
        ASSERT_EQ(t->agent_info->id, 1U);
        ASSERT_NE(t->process_info, nullptr);
        ASSERT_EQ(t->process_info->pid, 100U);
        ASSERT_NE(t->node_info, nullptr);
    }
}

TEST_F(reader_v4_kd_pmc_test, v4_kd_pmc_pmc_info_populated)
{
    auto tracks =
        find_tracks(m_reader->get_all_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_EQ(tracks.size(), 2U);

    std::set<std::string> pmc_names;
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->pmc_info, nullptr);
        pmc_names.insert(t->pmc_info->name);
    }
    ASSERT_TRUE(pmc_names.count("SQ_WAVES") == 1);
    ASSERT_TRUE(pmc_names.count("GRBM_COUNT") == 1);
}

TEST_F(reader_v4_kd_pmc_test, v4_kd_pmc_interval_track_count_and_order)
{
    // Timestamps inserted out of value order (kd 2 timestamps ids 1,2 with values
    // 2000/2300 before kd 1 timestamps ids 3,4 with values 1000/1200). ORDER BY
    // ts_s.value must return kd 1 before kd 2 on the SQ_WAVES track.
    auto tracks =
        find_tracks(m_reader->get_all_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_EQ(tracks.size(), 2U);

    profiler_hub::reader_types::track_info_ptr_t sq_waves_track;
    profiler_hub::reader_types::track_info_ptr_t grbm_track;
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->pmc_info, nullptr);
        if(t->pmc_info->name == "SQ_WAVES")
            sq_waves_track = t;
        else if(t->pmc_info->name == "GRBM_COUNT")
            grbm_track = t;
    }
    ASSERT_NE(sq_waves_track, nullptr);
    ASSERT_NE(grbm_track, nullptr);

    auto sq_intervals = m_reader->get_interval_track(sq_waves_track->id);
    ASSERT_EQ(sq_intervals.size(), 2U);
    ASSERT_TRUE(is_start_sorted(sq_intervals));
    ASSERT_EQ(sq_intervals[0].start, 1000U);
    ASSERT_EQ(sq_intervals[0].end, 1200U);
    ASSERT_EQ(sq_intervals[1].start, 2000U);
    ASSERT_EQ(sq_intervals[1].end, 2300U);

    auto grbm_intervals = m_reader->get_interval_track(grbm_track->id);
    ASSERT_EQ(grbm_intervals.size(), 1U);
    ASSERT_EQ(grbm_intervals[0].start, 3000U);
    ASSERT_EQ(grbm_intervals[0].end, 3100U);
}

TEST_F(reader_v4_kd_pmc_test, v4_kd_pmc_track_stats_matches_interval_slice)
{
    auto tracks =
        find_tracks(m_reader->get_all_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    for(const auto& t : tracks)
    {
        auto intervals = m_reader->get_interval_track(t->id);
        auto stats     = m_reader->get_track_stats(t->id);
        expect_stats_match_intervals(stats, intervals);
    }
}

TEST_F(reader_v4_kd_pmc_test, v4_kd_pmc_display_name_from_kernel_symbol)
{
    auto tracks =
        find_tracks(m_reader->get_all_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_GE(tracks.size(), 1U);
    auto intervals = m_reader->get_interval_track(tracks.front()->id);
    ASSERT_FALSE(intervals.empty());
    for(const auto& ev : intervals)
    {
        ASSERT_EQ(ev.display_name, "vecAdd(float*, int)");
    }
}

TEST_F(reader_v4_kd_pmc_test, v4_get_scalar_track_returns_empty_for_kd_pmc)
{
    auto tracks =
        find_tracks(m_reader->get_all_tracks(),
                    profiler_hub::reader_types::track_type_t::kernel_dispatch_pmc);
    ASSERT_GE(tracks.size(), 1U);
    ASSERT_TRUE(m_reader->get_scalar_track(tracks.front()->id).empty());
}

// ============================================================================
// memory_activity track type — v3 synthetic fixture (rocpd_v3_mem_activity.db)
// Covers: discovery, running-sum correctness (ALLOC/FREE/REALLOC/RECLAIM),
// FREE agent_id recovery via address self-join, non-interference between agents.
// ============================================================================

class reader_v3_mem_activity_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string m_database_path{ ROCPD_DB_V3_MEM_ACTIVITY_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_mem_activity_test, v3_discovers_two_mem_activity_tracks)
{
    // Two distinct (nid, pid, agent_id): agent 1 and agent 2.
    auto tracks = find_tracks(m_reader->get_all_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    ASSERT_EQ(tracks.size(), 2U);

    // Each track must carry agent_info; no pmc_info (fidelity caveat #2).
    for(const auto& t : tracks)
    {
        ASSERT_NE(t->agent_info, nullptr);
        ASSERT_EQ(t->pmc_info, nullptr);
        ASSERT_NE(t->node_info, nullptr);
        ASSERT_NE(t->process_info, nullptr);
    }

    std::set<size_t> agent_ids;
    for(const auto& t : tracks)
        agent_ids.insert(t->agent_info->id);
    ASSERT_TRUE(agent_ids.count(1) == 1);
    ASSERT_TRUE(agent_ids.count(2) == 1);
}

TEST_F(reader_v3_mem_activity_test, v3_mem_activity_running_sum_agent1)
{
    // Agent 1 series: ALLOC(4096) at ts=1000, FREE-recovered at ts=3000,
    // REALLOC(no-op) at ts=4000, ALLOC(2048) at ts=5000.
    // Expected 3 scalar samples (REALLOC is not emitted).
    auto tracks = find_tracks(m_reader->get_all_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    profiler_hub::reader_types::track_info_ptr_t agent1_track;
    for(const auto& t : tracks)
    {
        if(t->agent_info && t->agent_info->id == 1) agent1_track = t;
    }
    ASSERT_NE(agent1_track, nullptr);

    auto scalars = m_reader->get_scalar_track(agent1_track->id);
    ASSERT_EQ(scalars.size(), 3U);

    // Timestamps must be ascending.
    ASSERT_EQ(scalars[0].timestamp, 1000U);
    ASSERT_EQ(scalars[1].timestamp, 3000U);
    ASSERT_EQ(scalars[2].timestamp, 5000U);

    // Running-sum values.
    ASSERT_DOUBLE_EQ(scalars[0].value, 4096.0);  // ALLOC +4096
    ASSERT_DOUBLE_EQ(scalars[1].value, 0.0);     // FREE -4096 (recovered)
    ASSERT_DOUBLE_EQ(scalars[2].value, 2048.0);  // ALLOC +2048
}

TEST_F(reader_v3_mem_activity_test, v3_mem_activity_free_agent_recovery)
{
    // The FREE row (row 3) has agent_id=NULL in the DB. Its size and agent must be
    // recovered from the ALLOC at the same address (4096). The running sum for agent 1
    // goes from 4096 to 0 at ts=3000, proving the recovery was correct.
    auto tracks = find_tracks(m_reader->get_all_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    profiler_hub::reader_types::track_info_ptr_t agent1_track;
    for(const auto& t : tracks)
    {
        if(t->agent_info && t->agent_info->id == 1) agent1_track = t;
    }
    ASSERT_NE(agent1_track, nullptr);

    auto scalars = m_reader->get_scalar_track(agent1_track->id);
    ASSERT_GE(scalars.size(), 2U);
    // The second sample (ts=3000) reflects the FREE: cumsum drops to 0.
    ASSERT_EQ(scalars[1].timestamp, 3000U);
    ASSERT_DOUBLE_EQ(scalars[1].value, 0.0);
}

TEST_F(reader_v3_mem_activity_test, v3_mem_activity_non_interference_agent2)
{
    // Agent 2 has exactly 1 ALLOC (ts=2000, size=8192). Its scalar series must not
    // include any agent-1 rows (ALLOC/FREE/REALLOC) or the REALLOC no-op.
    auto tracks = find_tracks(m_reader->get_all_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    profiler_hub::reader_types::track_info_ptr_t agent2_track;
    for(const auto& t : tracks)
    {
        if(t->agent_info && t->agent_info->id == 2) agent2_track = t;
    }
    ASSERT_NE(agent2_track, nullptr);

    auto scalars = m_reader->get_scalar_track(agent2_track->id);
    ASSERT_EQ(scalars.size(), 1U);
    ASSERT_EQ(scalars[0].timestamp, 2000U);
    ASSERT_DOUBLE_EQ(scalars[0].value, 8192.0);
}

TEST_F(reader_v3_mem_activity_test, v3_mem_activity_track_stats)
{
    auto tracks = find_tracks(m_reader->get_all_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    for(const auto& t : tracks)
    {
        auto scalars = m_reader->get_scalar_track(t->id);
        auto stats   = m_reader->get_track_stats(t->id);
        ASSERT_TRUE(stats.min_ts.has_value());
        ASSERT_TRUE(stats.max_ts.has_value());
        ASSERT_EQ(stats.count, scalars.size());
        ASSERT_EQ(stats.min_ts.value(), scalars.front().timestamp);
        ASSERT_EQ(stats.max_ts.value(), scalars.back().timestamp);
    }
}

TEST_F(reader_v3_mem_activity_test, v3_get_interval_track_returns_empty_for_mem_activity)
{
    // memory_activity is a scalar-only track; interval read must return empty.
    auto tracks = find_tracks(m_reader->get_all_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    ASSERT_GE(tracks.size(), 1U);
    ASSERT_TRUE(m_reader->get_interval_track(tracks.front()->id).empty());
}

// ============================================================================
// memory_activity track type — v4.0 synthetic fixture (rocpd_v4_mem_activity.db)
// Mirrors the v3 fixture data shape; the presence of rocpd_timestamp triggers
// the v4 backend. agent_id comes from rocpd_track JOIN (no NULL agent needed).
// ============================================================================

class reader_v4_mem_activity_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string m_database_path{ ROCPD_DB_V4_MEM_ACTIVITY_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v4_mem_activity_test, v4_discovers_two_mem_activity_tracks)
{
    auto tracks = find_tracks(m_reader->get_all_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    ASSERT_EQ(tracks.size(), 2U);

    for(const auto& t : tracks)
    {
        ASSERT_NE(t->agent_info, nullptr);
        ASSERT_EQ(t->pmc_info, nullptr);
    }

    std::set<size_t> agent_ids;
    for(const auto& t : tracks)
        agent_ids.insert(t->agent_info->id);
    ASSERT_TRUE(agent_ids.count(1) == 1);
    ASSERT_TRUE(agent_ids.count(2) == 1);
}

TEST_F(reader_v4_mem_activity_test, v4_mem_activity_running_sum_agent1)
{
    // Same logical sequence as v3: ALLOC(4096)+FREE(4096)+REALLOC(no-op)+ALLOC(2048).
    // Rows inserted out of start order to prove ORDER BY ts_s.value.
    auto tracks = find_tracks(m_reader->get_all_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    profiler_hub::reader_types::track_info_ptr_t agent1_track;
    for(const auto& t : tracks)
    {
        if(t->agent_info && t->agent_info->id == 1) agent1_track = t;
    }
    ASSERT_NE(agent1_track, nullptr);

    auto scalars = m_reader->get_scalar_track(agent1_track->id);
    ASSERT_EQ(scalars.size(), 3U);

    ASSERT_EQ(scalars[0].timestamp, 1000U);
    ASSERT_EQ(scalars[1].timestamp, 3000U);
    ASSERT_EQ(scalars[2].timestamp, 5000U);

    ASSERT_DOUBLE_EQ(scalars[0].value, 4096.0);
    ASSERT_DOUBLE_EQ(scalars[1].value, 0.0);
    ASSERT_DOUBLE_EQ(scalars[2].value, 2048.0);
}

TEST_F(reader_v4_mem_activity_test, v4_mem_activity_non_interference_agent2)
{
    auto tracks = find_tracks(m_reader->get_all_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    profiler_hub::reader_types::track_info_ptr_t agent2_track;
    for(const auto& t : tracks)
    {
        if(t->agent_info && t->agent_info->id == 2) agent2_track = t;
    }
    ASSERT_NE(agent2_track, nullptr);

    auto scalars = m_reader->get_scalar_track(agent2_track->id);
    ASSERT_EQ(scalars.size(), 1U);
    ASSERT_EQ(scalars[0].timestamp, 2000U);
    ASSERT_DOUBLE_EQ(scalars[0].value, 8192.0);
}

TEST_F(reader_v4_mem_activity_test, v4_mem_activity_track_stats)
{
    auto tracks = find_tracks(m_reader->get_all_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    for(const auto& t : tracks)
    {
        auto scalars = m_reader->get_scalar_track(t->id);
        auto stats   = m_reader->get_track_stats(t->id);
        ASSERT_TRUE(stats.min_ts.has_value());
        ASSERT_TRUE(stats.max_ts.has_value());
        ASSERT_EQ(stats.count, scalars.size());
        ASSERT_EQ(stats.min_ts.value(), scalars.front().timestamp);
        ASSERT_EQ(stats.max_ts.value(), scalars.back().timestamp);
    }
}

TEST_F(reader_v4_mem_activity_test, v4_get_interval_track_returns_empty_for_mem_activity)
{
    auto tracks = find_tracks(m_reader->get_all_tracks(),
                              profiler_hub::reader_types::track_type_t::memory_activity);
    ASSERT_GE(tracks.size(), 1U);
    ASSERT_TRUE(m_reader->get_interval_track(tracks.front()->id).empty());
}

// =============================================================================
// Task 014: ambiguous-pmc detection tests
//
// Three fixture tiers:
//   1. reader_test (rocpd.db, v3)   — 2358 PMCs; pmc_id 2356 is the lone
//                                      ambiguous case (2 rocpd_pmc_event rows
//                                      per event_id, verified by task 005B-4).
//   2. reader_v3_amb_pmc_test       — minimal v3 synthetic; 2 PMCs (pmc_id 1
//                                      ambiguous, pmc_id 2 clean).
//   3. reader_v4_amb_pmc_test       — same shape on the v4 backend.
// =============================================================================

// --- Tier 1: main v3 rocpd.db fixture ----------------------------------------

TEST_F(reader_test, pmc_id_2356_is_flagged_ambiguous)
{
    auto                                       pmc_list = m_reader->get_all_pmc_info();
    profiler_hub::reader_types::pmc_info_ptr_t pmc_2356;
    for(const auto& p : pmc_list)
    {
        if(p->pmc_id == 2356)
        {
            pmc_2356 = p;
            break;
        }
    }
    ASSERT_NE(pmc_2356, nullptr) << "pmc_id 2356 not found in rocpd.db";
    EXPECT_TRUE(pmc_2356->ambiguous);
}

TEST_F(reader_test, all_other_pmc_ids_are_not_ambiguous)
{
    auto   pmc_list        = m_reader->get_all_pmc_info();
    size_t ambiguous_count = 0;
    for(const auto& p : pmc_list)
    {
        if(p->ambiguous) ++ambiguous_count;
    }
    // Exactly one ambiguous pmc_id in this DB (pmc_id 2356).
    EXPECT_EQ(ambiguous_count, 1U);
}

// --- Tier 2: synthetic v3 ambiguous-pmc fixture ------------------------------

class reader_v3_amb_pmc_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_V3_AMB_PMC_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v3_amb_pmc_test, v3_ambiguous_pmc_id_flagged)
{
    auto pmc_list = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_list.size(), 2U);

    profiler_hub::reader_types::pmc_info_ptr_t fault_pmc;
    profiler_hub::reader_types::pmc_info_ptr_t clean_pmc;
    for(const auto& p : pmc_list)
    {
        if(p->name == "FAULT_COUNT") fault_pmc = p;
        if(p->name == "CLEAN_COUNT") clean_pmc = p;
    }
    ASSERT_NE(fault_pmc, nullptr);
    ASSERT_NE(clean_pmc, nullptr);
    EXPECT_TRUE(fault_pmc->ambiguous);
    EXPECT_FALSE(clean_pmc->ambiguous);
}

TEST_F(reader_v3_amb_pmc_test, v3_exactly_one_ambiguous_pmc)
{
    auto   pmc_list        = m_reader->get_all_pmc_info();
    size_t ambiguous_count = 0;
    for(const auto& p : pmc_list)
    {
        if(p->ambiguous) ++ambiguous_count;
    }
    EXPECT_EQ(ambiguous_count, 1U);
}

// --- Tier 3: synthetic v4.0 ambiguous-pmc fixture ----------------------------

class reader_v4_amb_pmc_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_V4_AMB_PMC_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v4_amb_pmc_test, v4_ambiguous_pmc_id_flagged)
{
    auto pmc_list = m_reader->get_all_pmc_info();
    ASSERT_EQ(pmc_list.size(), 2U);

    profiler_hub::reader_types::pmc_info_ptr_t fault_pmc;
    profiler_hub::reader_types::pmc_info_ptr_t clean_pmc;
    for(const auto& p : pmc_list)
    {
        if(p->name == "FAULT_COUNT") fault_pmc = p;
        if(p->name == "CLEAN_COUNT") clean_pmc = p;
    }
    ASSERT_NE(fault_pmc, nullptr);
    ASSERT_NE(clean_pmc, nullptr);
    EXPECT_TRUE(fault_pmc->ambiguous);
    EXPECT_FALSE(clean_pmc->ambiguous);
}

TEST_F(reader_v4_amb_pmc_test, v4_exactly_one_ambiguous_pmc)
{
    auto   pmc_list        = m_reader->get_all_pmc_info();
    size_t ambiguous_count = 0;
    for(const auto& p : pmc_list)
    {
        if(p->ambiguous) ++ambiguous_count;
    }
    EXPECT_EQ(ambiguous_count, 1U);
}

// Task 018: v4 track-classification ambiguity detection tests
//
// Fixture: rocpd_v4_amb_cls.db — a single rocpd_track row (id=1) referenced by
//   both rocpd_sample/rocpd_pmc_event (counter set) and rocpd_memory_allocate
//   (memory set). build_v4_tracks() must detect the overlap, log a warning, and
//   set ambiguous_classification=true on the resulting counter track.

class reader_v4_amb_cls_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, "");
        m_reader  = std::make_shared<profiler_hub::reader_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_reader.reset();
        m_storage.reset();
    }

    std::string                              m_database_path{ ROCPD_DB_V4_AMB_CLS_PATH };
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::reader_t>  m_reader;
};

TEST_F(reader_v4_amb_cls_test, v4_ambiguous_classification_track_flagged)
{
    auto tracks = m_reader->get_all_tracks();

    // Find the counter track (the ambiguous rocpd_track row; the fixture also
    // yields a synthetic memory_activity track from the same rocpd_memory_allocate).
    profiler_hub::reader_types::track_info_ptr_t counter_track;
    for(const auto& t : tracks)
    {
        if(t->type == profiler_hub::reader_types::track_type_t::counter)
        {
            counter_track = t;
            break;
        }
    }
    ASSERT_NE(counter_track, nullptr) << "no counter track found";
    // Counter classification wins (existing precedence).
    EXPECT_EQ(counter_track->type, profiler_hub::reader_types::track_type_t::counter);
    // Overlap with memory-allocate set is detected and flagged.
    EXPECT_TRUE(counter_track->ambiguous_classification);
}

TEST_F(reader_v4_amb_cls_test, v4_non_ambiguous_track_not_flagged)
{
    // The main v4 fixture has no overlapping track_ids; no track should be flagged.
    auto storage = std::make_unique<profiler_hub::storage_t>(ROCPD_DB_V4_PATH, "");
    auto reader  = std::make_shared<profiler_hub::reader_t>(std::move(storage));

    for(const auto& t : reader->get_all_tracks())
    {
        EXPECT_FALSE(t->ambiguous_classification)
            << "unexpected ambiguous_classification on track id=" << t->id;
    }
}

}  // namespace
