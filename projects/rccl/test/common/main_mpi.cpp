/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file main_mpi.cpp
 * @brief Main entry point for Google Test-based MPI tests
 *
 * This file provides the main() function for running GTest-based MPI tests.
 * For standalone tests (performance benchmarks, etc.), each test should have
 * its own main() function and use MPIHelpers for common functionality.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>

#ifdef MPI_TESTS_ENABLED

    #include "MPIHelpers.hpp"
    #include "MPITestBase.hpp"
    #include "MPIEnvironment.hpp"

namespace
{
/**
 * @brief Parse and strip --net_ib_nthreads=N from argv before GTest sees it
 *
 * Mirrors rccl-tests' own -t/--nthreads handling (parsed in its hand-written
 * main() before any collective/threading logic runs): a custom flag must be
 * consumed here since GTest's InitGoogleTest() does not know about it.
 */
int parseAndStripNThreads(int* argc, char** argv)
{
    constexpr const char* kFlagPrefix = "--net_ib_nthreads=";
    const size_t           kPrefixLen = std::strlen(kFlagPrefix);
    int                    nThreads   = 1;

    int writeIdx = 1;
    for(int readIdx = 1; readIdx < *argc; ++readIdx)
    {
        if(std::strncmp(argv[readIdx], kFlagPrefix, kPrefixLen) == 0)
        {
            nThreads = std::atoi(argv[readIdx] + kPrefixLen);
            if(nThreads < 1) nThreads = 1;
            // Upper bound mirrors the MAX_THREADS convention already used for
            // NCCL_SOCKET_NTHREADS in NetSocketTests.cpp: a typo'd huge value
            // should degrade to something runnable, not exhaust the machine.
            if(nThreads > MPIEnvironment::kMaxThreads)
            {
                std::fprintf(stderr,
                             "WARNING: --net_ib_nthreads=%d exceeds maximum %d, clamping\n",
                             nThreads,
                             MPIEnvironment::kMaxThreads);
                nThreads = MPIEnvironment::kMaxThreads;
            }
            continue;
        }
        argv[writeIdx++] = argv[readIdx];
    }
    *argc = writeIdx;
    // C/POSIX guarantee argv[argc] == NULL; MPI_Init_thread() receives this
    // argv, so restore the terminator after compacting.
    argv[writeIdx] = nullptr;
    return nThreads;
}
} // namespace

int main(int argc, char* argv[])
{
    // Parse our own custom flag before anything else touches argv
    MPIEnvironment::nThreads = parseAndStripNThreads(&argc, argv);

    // Initialize MPI using shared helper
    auto mpi_ctx = MPIHelpers::initializeMPI(&argc, &argv);

    // Record what MPI actually granted: initializeMPI() *requests*
    // MPI_THREAD_MULTIPLE but a given MPI build may provide less, and the
    // multithread test paths need it to legally issue MPI calls off-thread.
    MPIEnvironment::mpiThreadSupport = mpi_ctx.thread_support;

    const auto world_rank = mpi_ctx.world_rank;
    const auto world_size = mpi_ctx.world_size;

    // Setup per-rank logging using shared helper
    auto       rank_log_config          = MPIHelpers::setupRankLogging(world_rank);
    const auto per_rank_logging_enabled = rank_log_config && rank_log_config->logging_enabled;

    // Print initialization message
    if(world_rank == 0 && !per_rank_logging_enabled)
    {
        TEST_INFO("MPI initialized - World size: %d, Thread support: %d",
                  world_size,
                  mpi_ctx.thread_support);
    }

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Suppress GTest output for non-zero ranks (unless per-rank logging is enabled)
    // This is done by deleting GTest listeners for non-zero ranks
    // Note: stdout/stderr are already redirected for non-zero ranks by setupRankLogging
    if(world_rank != 0 && !per_rank_logging_enabled)
    {
        auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
        delete listeners.Release(listeners.default_xml_generator());
    }

    // Set up the RCCL MPI environment for all tests
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment());

    // Run all tests
    const auto ret_code = RUN_ALL_TESTS();

    // Restore original output if per-rank logging was enabled
    if(rank_log_config)
    {
        MPIHelpers::restoreRankLogging(*rank_log_config);
    }

    // MPI_Finalize is called by:
    // 1. MPIEnvironment::TearDown() -> cleanup_mpi() (normal case)
    // 2. MPIEnvironment destructor (safety net if TearDown fails or no tests match)
    return ret_code;
}

#else // MPI_TESTS_ENABLED not defined

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    std::fprintf(stderr,
                 "ERROR: MPI tests are not enabled. Please build with ENABLE_MPI_TESTS=ON\n");
    std::fprintf(stderr, "Usage: cmake -DENABLE_MPI_TESTS=ON -DMPI_PATH=/path/to/mpi ..\n");
    return 1;
}

#endif // MPI_TESTS_ENABLED
