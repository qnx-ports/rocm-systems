// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/reader.hpp"
#include "profiler-hub/storage.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <string>

namespace
{

// --------------------------------------------------------------------------
// Track-scoped API test helpers (shared by v3 / v4 / v4-counter fixtures).
// --------------------------------------------------------------------------

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
    // (rows referenced by rocpd_sample). Synthesis adds 1 gpu_queue + 1 dma + 1 region
    // (the sole (nid,pid,tid)=(...,67979,1) thread, all regions main => one track):
    //   54 counter + 1 gpu_queue + 1 dma + 1 cpu_thread = 57.
    ASSERT_EQ(track_list.size(), 57);
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
    ASSERT_EQ(first.opaque_id, 59);
    ASSERT_EQ(first.start, 23040260707644);
    ASSERT_EQ(first.end, 23040498732102);
    ASSERT_GE(first.end, first.start);

    auto details = m_reader->get_region_details(first.opaque_id);
    ASSERT_TRUE(details.has_value());
    ASSERT_EQ(details->name, "bit_extract");
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

    // First sample's value is reproducible via get_scalar_details() on its opaque id.
    const auto& first   = samples.front();
    auto        details = m_reader->get_scalar_details(first.opaque_id);
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
    // + 0 region->memory_allocate = 3 total (stack_id linkage).
    auto flows = m_reader->get_flows();
    ASSERT_EQ(flows.size(), 3);
    for(const auto& f : flows)
    {
        ASSERT_GT(f.source_opaque_id, 0U);
        ASSERT_GT(f.dest_opaque_id, 0U);
    }
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
    // rocpd_track contributes only its 3 sampled (counter) rows (2, 3, 6); the
    // non-counter rows (1, 4, 5) are ignored. Synthesis adds 1 cpu_thread (the sole
    // (1,1,1) thread, all regions main), 2 gpu_queue, 2 dma => 8 tracks total.
    auto tracks = m_reader->get_all_tracks();
    ASSERT_EQ(tracks.size(), 8U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::cpu_thread).size(),
        1U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter).size(),
        3U);
    ASSERT_EQ(
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::gpu_queue).size(),
        2U);
    ASSERT_EQ(find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma).size(),
              2U);
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
    auto tracks = m_reader->get_all_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_EQ(counters.size(), 3U);

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
    // Exactly one carries no process (pid NULL -- track 6); the other two do.
    ASSERT_EQ(without_process, 1);
    ASSERT_EQ(with_process, 2);
}

TEST_F(reader_v3_edge_test, counter_thread_info_tracks_tid_agent_info_always_null)
{
    // The #147 contract, both branches. thread_info is driven by rocpd_track.tid
    // and is orthogonal to counter classification; agent_info is impossible on v3
    // (rocpd_track has no agent_id column) regardless of tid.
    auto tracks = m_reader->get_all_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);
    ASSERT_EQ(counters.size(), 3U);

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
    ASSERT_EQ(regions.front().opaque_id, 2U);
    ASSERT_EQ(regions.front().start, 1000);
    ASSERT_EQ(regions.front().end, 5000);

    auto details = m_reader->get_region_details(regions.front().opaque_id);
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

    // Two dma tracks: Stream-X has 2 copies (start 2100, 2200), Stream-Y 1.
    auto dma = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    ASSERT_EQ(dma.size(), 2U);
    profiler_hub::reader_types::interval_event_list_t dma_two;
    size_t                                            dma_singletons = 0;
    for(const auto& t : dma)
    {
        auto iv = m_reader->get_interval_track(t->id);
        if(iv.size() == 2)
            dma_two = iv;
        else if(iv.size() == 1)
            ++dma_singletons;
    }
    ASSERT_EQ(dma_two.size(), 2U);
    ASSERT_EQ(dma_singletons, 1U);
    ASSERT_TRUE(is_start_sorted(dma_two));
    ASSERT_EQ(dma_two.front().start, 2100);
}

TEST_F(reader_v3_edge_test, get_scalar_track_values_for_both_counters)
{
    auto tracks = m_reader->get_all_tracks();
    auto counters =
        find_tracks(tracks, profiler_hub::reader_types::track_type_t::counter);

    for(const auto& c : counters)
    {
        auto samples = m_reader->get_scalar_track(c->id);
        ASSERT_FALSE(samples.empty());
        ASSERT_TRUE(is_timestamp_sorted(samples));

        if(c->name == "GRBM_COUNT")
        {
            // 3 samples, ascending timestamp despite differing row-id order.
            ASSERT_EQ(samples.size(), 3U);
            ASSERT_EQ(samples.front().timestamp, 1000);
            ASSERT_DOUBLE_EQ(samples.front().value, 10.5);
        }
        else if(c->name == "SQ_WAVES")
        {
            ASSERT_EQ(samples.size(), 2U);
            ASSERT_EQ(samples.front().timestamp, 500);
            ASSERT_DOUBLE_EQ(samples.front().value, 5.0);
        }

        // Value is reproducible via get_scalar_details() on the opaque id.
        auto details = m_reader->get_scalar_details(samples.front().opaque_id);
        ASSERT_TRUE(details.has_value());
        ASSERT_DOUBLE_EQ(details->value, samples.front().value);
    }
}

TEST_F(reader_v3_edge_test, get_flows_excludes_zero_and_null_stack_id)
{
    // stack_id linkage (Q4): region<->kernel_dispatch (100), region<->memory_copy
    // (200), region<->memory_allocate (400) = 3 flows. RegionGamma (stack 0) and
    // the sample events (stack NULL) are excluded.
    auto flows = m_reader->get_flows();
    ASSERT_EQ(flows.size(), 3U);
    for(const auto& f : flows)
    {
        ASSERT_GT(f.source_opaque_id, 0U);
        ASSERT_GT(f.dest_opaque_id, 0U);
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
    // Fixture has 4 tracks: 1 cpu_thread, 1 gpu_queue, 2 dma.
    ASSERT_EQ(tracks.size(), 4);

    auto cpu = find_tracks(tracks, profiler_hub::reader_types::track_type_t::cpu_thread);
    auto gpu = find_tracks(tracks, profiler_hub::reader_types::track_type_t::gpu_queue);
    auto dma = find_tracks(tracks, profiler_hub::reader_types::track_type_t::dma);
    ASSERT_EQ(cpu.size(), 1);
    ASSERT_EQ(gpu.size(), 1);
    ASSERT_EQ(dma.size(), 2);

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
    ASSERT_GT(first.opaque_id, 0U);

    // opaque_id resolves through the region detail path.
    ASSERT_TRUE(m_reader->get_region_details(first.opaque_id).has_value());
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

    // opaque_id resolves through the kernel dispatch detail path.
    ASSERT_TRUE(
        m_reader->get_kernel_dispatch_details(intervals.front().opaque_id).has_value());
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
        ASSERT_TRUE(
            m_reader->get_memory_copy_details(intervals.front().opaque_id).has_value());
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
    auto flows = m_reader->get_flows();
    ASSERT_EQ(flows.size(), 22);
    for(const auto& f : flows)
    {
        ASSERT_GT(f.source_opaque_id, 0U);
        ASSERT_GT(f.dest_opaque_id, 0U);
    }
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

    ASSERT_EQ(samples[0].opaque_id, 2);
    ASSERT_EQ(samples[0].timestamp, 1000);
    ASSERT_DOUBLE_EQ(samples[0].value, 10.5);

    ASSERT_EQ(samples[1].opaque_id, 3);
    ASSERT_EQ(samples[1].timestamp, 2000);
    ASSERT_DOUBLE_EQ(samples[1].value, 20.5);

    ASSERT_EQ(samples[2].opaque_id, 1);
    ASSERT_EQ(samples[2].timestamp, 3000);
    ASSERT_DOUBLE_EQ(samples[2].value, 30.5);
}

TEST_F(reader_v4_counter_test, v4_get_scalar_details_resolves_value_and_timestamp)
{
    // opaque_id 1 -> sample id 1 -> timestamp 3000, value 30.5.
    auto details = m_reader->get_scalar_details(1);
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

}  // namespace
