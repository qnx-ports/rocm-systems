/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_process.hh>
#include <utils.hh>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "hipGetProcAddressHelpers.hh"

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - memory IPC related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress_IPC_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipGetProcAddress_IPC_Memory) {
  int N = 40;
  int Nbytes = N * sizeof(int);

  int fd[2];
  REQUIRE(pipe(fd) == 0);

  auto pid = fork();

  // Validating hipIpcGetMemHandle API
  if (pid != 0) {  // parent process
    void* hipIpcGetMemHandle_ptr = nullptr;

    int currentHipVersion = 0;
    HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

    HIP_CHECK(hipGetProcAddress("hipIpcGetMemHandle", &hipIpcGetMemHandle_ptr, currentHipVersion, 0,
                                nullptr));

    hipError_t (*dyn_hipIpcGetMemHandle_ptr)(hipIpcMemHandle_t*, void*) =
        reinterpret_cast<hipError_t (*)(hipIpcMemHandle_t*, void*)>(hipIpcGetMemHandle_ptr);

    int* srcHostMem = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(srcHostMem != nullptr);
    fillHostArray(srcHostMem, N, 10);

    int* devMemSrc = nullptr;
    HIP_CHECK(hipMalloc(&devMemSrc, Nbytes));
    REQUIRE(devMemSrc != nullptr);
    HIP_CHECK(hipMemcpy(devMemSrc, srcHostMem, Nbytes, hipMemcpyHostToDevice));

    hipIpcMemHandle_t handle;
    HIP_CHECK(dyn_hipIpcGetMemHandle_ptr(&handle, devMemSrc));

    REQUIRE(hip::writeAll(fd[1], &handle, sizeof(handle)));
    REQUIRE(close(fd[1]) == 0);

    REQUIRE(wait(NULL) >= 0);

    HIP_CHECK(hipFree(devMemSrc));
    free(srcHostMem);
  } else {  // child process
    // Validating hipIpcOpenMemHandle, hipIpcCloseMemHandle API's
    void* hipIpcOpenMemHandle_ptr = nullptr;
    void* hipIpcCloseMemHandle_ptr = nullptr;

    int currentHipVersion = 0;
    HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

    HIP_CHECK(hipGetProcAddress("hipIpcOpenMemHandle", &hipIpcOpenMemHandle_ptr, currentHipVersion,
                                0, nullptr));
    HIP_CHECK(hipGetProcAddress("hipIpcCloseMemHandle", &hipIpcCloseMemHandle_ptr,
                                currentHipVersion, 0, nullptr));

    hipError_t (*dyn_hipIpcOpenMemHandle_ptr)(void**, hipIpcMemHandle_t, unsigned int) =
        reinterpret_cast<hipError_t (*)(void**, hipIpcMemHandle_t, unsigned int)>(
            hipIpcOpenMemHandle_ptr);
    hipError_t (*dyn_hipIpcCloseMemHandle_ptr)(void*) =
        reinterpret_cast<hipError_t (*)(void*)>(hipIpcCloseMemHandle_ptr);

    hipIpcMemHandle_t handle;
    REQUIRE(hip::readAll(fd[0], &handle, sizeof(handle)));
    REQUIRE(close(fd[0]) == 0);

    int* devPtr = nullptr;
    HIP_CHECK(dyn_hipIpcOpenMemHandle_ptr(reinterpret_cast<void**>(&devPtr), handle,
                                          hipIpcMemLazyEnablePeerAccess));
    REQUIRE(devPtr != nullptr);

    addOneKernel<<<1, 1>>>(devPtr, N);

    int* dstHostMem = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(dstHostMem != nullptr);

    HIP_CHECK(hipMemcpy(dstHostMem, devPtr, Nbytes, hipMemcpyDeviceToHost));
    REQUIRE(validateHostArray(dstHostMem, N, 11) == true);

    HIP_CHECK(dyn_hipIpcCloseMemHandle_ptr(devPtr));

    free(dstHostMem);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Verifies that hipGetProcAddress returns usable function pointers for the
 *  - Event IPC APIs (hipIpcGetEventHandle / hipIpcOpenEventHandle): the parent
 *  - exports an interprocess event handle through the resolved
 *  - hipIpcGetEventHandle pointer, and the child imports it through the resolved
 *  - hipIpcOpenEventHandle pointer in a separate process.
 *  - This test's unique purpose is to exercise the proc-address dispatch path for
 *  - these APIs (address + ABI). Full cross-process event *synchronization*
 *  - semantics are covered by Unit_hipIpcEventHandle_Functional.
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress_IPC_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipGetProcAddress_IPC_Event) {
  // Enable gated IPC-event tracing in the HIP runtime for both processes. Set
  // before fork()/HIP init so the child inherits it.
  setenv("HIP_IPC_EVENT_DEBUG", "1", 1);

  int fd[2];
  REQUIRE(pipe(fd) == 0);

  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid != 0) {  // parent process: exports the interprocess event handle
    void* hipIpcGetEventHandle_ptr = nullptr;

    int currentHipVersion = 0;
    HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

    HIP_CHECK(hipGetProcAddress("hipIpcGetEventHandle", &hipIpcGetEventHandle_ptr,
                                currentHipVersion, 0, nullptr));
    REQUIRE(hipIpcGetEventHandle_ptr != nullptr);

    auto dyn_hipIpcGetEventHandle_ptr =
        reinterpret_cast<hipError_t (*)(hipIpcEventHandle_t*, hipEvent_t)>(
            hipIpcGetEventHandle_ptr);

    // Interprocess events must disable timing.
    hipEvent_t event = nullptr;
    HIP_CHECK(hipEventCreateWithFlags(&event, hipEventInterprocess | hipEventDisableTiming));
    REQUIRE(event != nullptr);

    hipIpcEventHandle_t handle{};
    HIP_CHECK(dyn_hipIpcGetEventHandle_ptr(&handle, event));

    // The parent only writes to the pipe.
    REQUIRE(close(fd[0]) == 0);

    REQUIRE(hip::writeAll(fd[1], &handle, sizeof(handle)));
    REQUIRE(close(fd[1]) == 0);

    // Surface any consumer-side failure (a forked child's REQUIREs do not
    // propagate to this process).
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == EXIT_SUCCESS);

    HIP_CHECK(hipEventDestroy(event));

  } else { // child process: imports the event handle in a separate process
    bool ok = false;
    try {
      void* hipIpcOpenEventHandle_ptr = nullptr;

      int currentHipVersion = 0;
      HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

      HIP_CHECK(hipGetProcAddress("hipIpcOpenEventHandle", &hipIpcOpenEventHandle_ptr,
                                  currentHipVersion, 0, nullptr));
      REQUIRE(hipIpcOpenEventHandle_ptr != nullptr);

      auto dyn_hipIpcOpenEventHandle_ptr =
          reinterpret_cast<hipError_t (*)(hipEvent_t*, hipIpcEventHandle_t)>(
              hipIpcOpenEventHandle_ptr);

      // The child only reads from the pipe.
      REQUIRE(close(fd[1]) == 0);

      hipIpcEventHandle_t handle{};
      REQUIRE(hip::readAll(fd[0], &handle, sizeof(handle)));
      REQUIRE(close(fd[0]) == 0);

      // The call under test: import the handle exported by the parent through the
      // proc-address-resolved pointer.
      fprintf(stderr, "[IPC_Event child] importing event handle\n");
      hipEvent_t event = nullptr;
      HIP_CHECK(dyn_hipIpcOpenEventHandle_ptr(&event, handle));
      REQUIRE(event != nullptr);

      // Exercise the imported event in a real GPU workload. NOTE: recording an
      // *imported* interprocess event (below) is what triggers the CI GPU
      // memory-access fault + hang under investigation. Per the HIP/CUDA
      // contract hipEventRecord "may be used in either process", so this should
      // be legal; Unit_hipGetProcAddress_IPC_Event_ParentRecord is the control
      // where the owner records instead.
      constexpr int N = 40;
      constexpr int Nbytes = N * sizeof(int);

      hipStream_t stream = nullptr;
      HIP_CHECK(hipStreamCreate(&stream));

      std::vector<int> hostMem(N, 10);

      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);

      HIP_CHECK(hipMemcpyAsync(devMem, hostMem.data(), Nbytes, hipMemcpyHostToDevice, stream));
      addOneKernel<<<1, 1, 0, stream>>>(devMem, N);
      HIP_CHECK(hipMemcpyAsync(hostMem.data(), devMem, Nbytes, hipMemcpyDeviceToHost, stream));

      // Record the imported event on the same stream and wait on it.
      fprintf(stderr, "[IPC_Event child] recording IMPORTED event (suspect)\n");
      HIP_CHECK(hipEventRecord(event, stream));
      HIP_CHECK(hipEventSynchronize(event));

      REQUIRE(validateHostArray(hostMem.data(), N, 11));

      HIP_CHECK(hipFree(devMem));
      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipEventDestroy(event));
      fprintf(stderr, "[IPC_Event child] success\n");
      ok = true;
    } catch (const std::exception& e) {
      // A forked child's REQUIRE/HIP_CHECK failures do not reach the Catch2
      // reporter, so log the reason to stderr before signalling via the exit
      // code -- otherwise CI only shows an opaque "Subprocess aborted".
      fprintf(stderr, "[IPC_Event child] failure: %s\n", e.what());
      _exit(EXIT_FAILURE);
    } catch (...) {
      fprintf(stderr, "[IPC_Event child] failure: unknown exception\n");
      _exit(EXIT_FAILURE);
    }
    _exit(ok ? EXIT_SUCCESS : EXIT_FAILURE);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Debug/control variant of Unit_hipGetProcAddress_IPC_Event used to test the
 *  - hypothesis that recording an *imported* interprocess event is what faults
 *  - the GPU. Here the OWNER (parent) records the event -- the canonical
 *  - direction (owner records, importer waits) -- while the child imports the
 *  - handle and only *waits* on it via hipStreamWaitEvent. If this passes while
 *  - Unit_hipGetProcAddress_IPC_Event (child records) faults, the failing path
 *  - is isolated to record-on-imported-event.
 *  - Sets HIP_IPC_EVENT_DEBUG so the runtime emits gated IPC-event tracing.
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress_IPC_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipGetProcAddress_IPC_Event_ParentRecord) {
  // Enable gated IPC-event tracing in the HIP runtime for both processes. Set
  // before fork()/HIP init so the child inherits it.
  setenv("HIP_IPC_EVENT_DEBUG", "1", 1);

  int fd[2];       // parent -> child: the exported event handle
  int fdReady[2];  // parent -> child: 1 byte signalling "event recorded"
  REQUIRE(pipe(fd) == 0);
  REQUIRE(pipe(fdReady) == 0);

  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid != 0) {  // parent process: owns AND records the interprocess event
    void* hipIpcGetEventHandle_ptr = nullptr;

    int currentHipVersion = 0;
    HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

    HIP_CHECK(hipGetProcAddress("hipIpcGetEventHandle", &hipIpcGetEventHandle_ptr,
                                currentHipVersion, 0, nullptr));
    REQUIRE(hipIpcGetEventHandle_ptr != nullptr);

    auto dyn_hipIpcGetEventHandle_ptr =
        reinterpret_cast<hipError_t (*)(hipIpcEventHandle_t*, hipEvent_t)>(
            hipIpcGetEventHandle_ptr);

    hipEvent_t event = nullptr;
    HIP_CHECK(hipEventCreateWithFlags(&event, hipEventInterprocess | hipEventDisableTiming));
    REQUIRE(event != nullptr);

    hipIpcEventHandle_t handle{};
    HIP_CHECK(dyn_hipIpcGetEventHandle_ptr(&handle, event));

    // The parent only writes to the pipes.
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdReady[0]) == 0);

    REQUIRE(hip::writeAll(fd[1], &handle, sizeof(handle)));
    REQUIRE(close(fd[1]) == 0);

    // Issue a real GPU workload and record the OWNED event on that stream.
    constexpr int N = 40;
    constexpr int Nbytes = N * sizeof(int);

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));

    std::vector<int> hostMem(N, 10);

    int* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);

    HIP_CHECK(hipMemcpyAsync(devMem, hostMem.data(), Nbytes, hipMemcpyHostToDevice, stream));
    addOneKernel<<<1, 1, 0, stream>>>(devMem, N);
    HIP_CHECK(hipMemcpyAsync(hostMem.data(), devMem, Nbytes, hipMemcpyDeviceToHost, stream));

    // Owner records the event -- the supported direction.
    fprintf(stderr, "[IPC_Event_ParentRecord parent] recording OWNED event\n");
    HIP_CHECK(hipEventRecord(event, stream));

    // Tell the child the event has been recorded so its hipStreamWaitEvent
    // observes a non-empty event state.
    const char ready = 1;
    REQUIRE(hip::writeAll(fdReady[1], &ready, sizeof(ready)));
    REQUIRE(close(fdReady[1]) == 0);

    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == EXIT_SUCCESS);

    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(validateHostArray(hostMem.data(), N, 11));

    HIP_CHECK(hipFree(devMem));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipEventDestroy(event));

  } else {  // child process: imports the event and only WAITS on it
    bool ok = false;
    try {
      void* hipIpcOpenEventHandle_ptr = nullptr;

      int currentHipVersion = 0;
      HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

      HIP_CHECK(hipGetProcAddress("hipIpcOpenEventHandle", &hipIpcOpenEventHandle_ptr,
                                  currentHipVersion, 0, nullptr));
      REQUIRE(hipIpcOpenEventHandle_ptr != nullptr);

      auto dyn_hipIpcOpenEventHandle_ptr =
          reinterpret_cast<hipError_t (*)(hipEvent_t*, hipIpcEventHandle_t)>(
              hipIpcOpenEventHandle_ptr);

      REQUIRE(close(fd[1]) == 0);
      REQUIRE(close(fdReady[1]) == 0);

      hipIpcEventHandle_t handle{};
      REQUIRE(hip::readAll(fd[0], &handle, sizeof(handle)));
      REQUIRE(close(fd[0]) == 0);

      fprintf(stderr, "[IPC_Event_ParentRecord child] importing event handle\n");
      hipEvent_t event = nullptr;
      HIP_CHECK(dyn_hipIpcOpenEventHandle_ptr(&event, handle));
      REQUIRE(event != nullptr);

      hipStream_t stream = nullptr;
      HIP_CHECK(hipStreamCreate(&stream));

      // Wait until the parent has recorded the event.
      char ready = 0;
      REQUIRE(hip::readAll(fdReady[0], &ready, sizeof(ready)));
      REQUIRE(close(fdReady[0]) == 0);

      // Importer only WAITS on the imported event -- the supported direction.
      fprintf(stderr, "[IPC_Event_ParentRecord child] waiting on imported event\n");
      HIP_CHECK(hipStreamWaitEvent(stream, event, 0));
      HIP_CHECK(hipStreamSynchronize(stream));

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipEventDestroy(event));
      fprintf(stderr, "[IPC_Event_ParentRecord child] success\n");
      ok = true;
    } catch (const std::exception& e) {
      fprintf(stderr, "[IPC_Event_ParentRecord child] failure: %s\n", e.what());
      _exit(EXIT_FAILURE);
    } catch (...) {
      fprintf(stderr, "[IPC_Event_ParentRecord child] failure: unknown exception\n");
      _exit(EXIT_FAILURE);
    }
    _exit(ok ? EXIT_SUCCESS : EXIT_FAILURE);
  }
}
