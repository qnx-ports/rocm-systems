## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

set(target test-roctx-recordfn)

add_executable(${target} test_roctx_recordfn.cpp)

# Includes, torch libraries, and the DebugInfoKind flag are inherited from
# roctx-recordfn. Only the runtime search paths below are test-specific.
target_link_libraries(${target} PRIVATE roctx-recordfn gtest_main)

# torch's shared libraries cross-reference one another at load time.
target_link_options(${target} PRIVATE LINKER:--allow-shlib-undefined)

if(INSTALL_TESTS)
    install(
        TARGETS ${target}
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${CMAKE_PROJECT_NAME}/tests
            COMPONENT tests
    )
endif()
