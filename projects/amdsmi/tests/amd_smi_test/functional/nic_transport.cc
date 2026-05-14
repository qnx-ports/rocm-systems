// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * NIC transport tests. The Parse and Consumer groups are hermetic (no device,
 * no syscall) and always run. The Hardware group exercises the real backend and
 * self-skips when no NIC is present, so the same binary covers both.
 */

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "smi_nic.h"
#include "smi_nic_system.h"
#include "smi_nic_transport.h"

namespace {

using amd::smi::nic::transport::DriverInfo;
using amd::smi::nic::transport::LinkSettings;
using amd::smi::nic::transport::NicTransport;
using amd::smi::nic::transport::parse_perm_addr;
using amd::smi::nic::transport::PauseParams;
using amd::smi::nic::transport::PermanentAddress;
using amd::smi::nic::transport::Result;
using amd::smi::nic::transport::VendorStatistics;

// Hermetic: pure perm-addr parser (guards the size-check regression)

TEST(NicTransportParse, ZeroSizeFails) {
  const uint8_t bytes[6] = {0, 0, 0, 0, 0, 0};
  auto r = parse_perm_addr(0, bytes);
  EXPECT_FALSE(r.success);
  EXPECT_EQ(r.error_code, ENODATA);
}

TEST(NicTransportParse, ShortSizeFails) {
  const uint8_t bytes[6] = {1, 2, 3, 4, 5, 6};
  auto r = parse_perm_addr(4, bytes);
  EXPECT_FALSE(r.success);
  EXPECT_EQ(r.error_code, ENODATA);
}

TEST(NicTransportParse, OversizeFails) {
  // The guard is !=, not <; a kernel reporting more than 6 bytes is rejected too.
  const uint8_t bytes[6] = {1, 2, 3, 4, 5, 6};
  auto r = parse_perm_addr(8, bytes);
  EXPECT_FALSE(r.success);
  EXPECT_EQ(r.error_code, ENODATA);
}

TEST(NicTransportParse, SixBytesSucceeds) {
  const uint8_t bytes[6] = {0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
  auto r = parse_perm_addr(6, bytes);
  ASSERT_TRUE(r.success);
  EXPECT_EQ(r.value.mac[0], 0xaa);
  EXPECT_EQ(r.value.mac[5], 0xff);
}

TEST(NicTransportParse, SixZeroBytesSucceeds) {
  /**
   * The size==6 contract is size-only: an all-zero MAC is a valid parse. The
   * "fail rather than fabricate zeros" rule applies to wrong sizes, not values.
   */
  const uint8_t bytes[6] = {0, 0, 0, 0, 0, 0};
  EXPECT_TRUE(parse_perm_addr(6, bytes).success);
}

// Hermetic: SmiNicPort consumer logic over an injected fake transport

/**
 * Returns canned Results so the port's formatting / optional-mapping can be
 * exercised without a device. Fields default to failure; set the ones a test
 * needs.
 */
class FakeTransport : public NicTransport {
 public:
  Result<PauseParams> pause{false, {}, ENODATA};
  Result<LinkSettings> link{false, {}, ENODATA};
  Result<PermanentAddress> perm{false, {}, ENODATA};

  Result<PauseParams> get_pause_params(const std::string&) override { return pause; }
  Result<LinkSettings> get_link_settings(const std::string&) override { return link; }
  Result<DriverInfo> get_driver_info(const std::string&) override { return {false, {}, ENODATA}; }
  Result<VendorStatistics> get_statistics(const std::string&) override { return {false, {}, ENODATA}; }
  Result<PermanentAddress> get_permanent_address(const std::string&) override { return perm; }
  std::string backend_name() const override { return "fake"; }
};

SmiNicPort make_fake_port(std::shared_ptr<NicTransport> transport) {
  /**
   * The sysfs paths are intentionally bogus; the ctor reads them via
   * get_sysfs_data, which returns nullopt on a missing path (no throw).
   */
  return SmiNicPort("eth-test", "0000:00:00.0", "/nonexistent/class", "/nonexistent/bus",
                    std::move(transport));
}

TEST(NicTransportConsumer, PermanentAddressFormatsMac) {
  auto fake = std::make_shared<FakeTransport>();
  PermanentAddress pa;
  /**
   * Mix hex-letter nibbles and a byte needing a leading zero (0x01) to catch
   * width/case regressions in the formatter.
   */
  pa.mac = {0xab, 0xcd, 0xef, 0x01, 0x23, 0x45};
  fake->perm = {true, pa, 0};

  auto port = make_fake_port(fake);
  auto addr = port.permanent_address();
  ASSERT_TRUE(addr.has_value());
  EXPECT_EQ(*addr, "ab:cd:ef:01:23:45");
}

TEST(NicTransportConsumer, PermanentAddressNulloptOnFailure) {
  auto fake = std::make_shared<FakeTransport>();  // perm defaults to failure
  auto port = make_fake_port(fake);
  EXPECT_FALSE(port.permanent_address().has_value());
}

TEST(NicTransportConsumer, AutonegMapsFromLinkSettings) {
  auto fake = std::make_shared<FakeTransport>();
  fake->link = {true, {}, 0};
  fake->link.value.autoneg = 1;

  auto port = make_fake_port(fake);
  auto an = port.autoneg();
  ASSERT_TRUE(an.has_value());
  EXPECT_TRUE(*an);
}

TEST(NicTransportConsumer, AutonegFalseFromLinkSettings) {
  auto fake = std::make_shared<FakeTransport>();
  fake->link = {true, {}, 0};
  fake->link.value.autoneg = 0;

  auto port = make_fake_port(fake);
  auto an = port.autoneg();
  ASSERT_TRUE(an.has_value());
  EXPECT_FALSE(*an);
}

TEST(NicTransportConsumer, AutonegNulloptOnFailure) {
  auto fake = std::make_shared<FakeTransport>();  // link defaults to failure
  auto port = make_fake_port(fake);
  EXPECT_FALSE(port.autoneg().has_value());
}

TEST(NicTransportConsumer, PauseParamsPassThrough) {
  auto fake = std::make_shared<FakeTransport>();
  fake->pause = {true, {}, 0};
  fake->pause.value.rx_pause = true;
  fake->pause.value.tx_pause = false;

  auto port = make_fake_port(fake);
  auto p = port.pause_params();
  ASSERT_TRUE(p.has_value());
  EXPECT_TRUE(p->rx_pause);
  EXPECT_FALSE(p->tx_pause);
}

// Hardware: real backend against a live NIC; skipped when none present

// Discovers once in SetUp and skips the whole group when no device is present.
class NicTransportHardware : public ::testing::Test {
 protected:
  void SetUp() override {
    system_.discover_nics();  // ctor registers the default vendor plugins
    if (system_.get_nics().empty()) {
      GTEST_SKIP() << "no NIC present";
    }
  }
  SmiNicSystem system_;
};

TEST_F(NicTransportHardware, RealBackendPortQueries) {
  const auto& nics = system_.get_nics();
  ASSERT_FALSE(nics.empty());

  const auto& ports = nics[0]->nic_ports();
  if (ports.empty()) {
    GTEST_SKIP() << "NIC has no ports";
  }
  const SmiNicPort& port = ports[0];

  /**
   * The point is exercising the real transport (and the C-ABI exception path it
   * feeds) without crashing; exact values are device-specific. A permanent
   * address, if present, must be a well-formed MAC rather than fabricated zeros.
   */
  auto addr = port.permanent_address();
  if (addr.has_value()) {
    EXPECT_EQ(addr->size(), 17u);
    EXPECT_NE(*addr, "00:00:00:00:00:00");
  }
  EXPECT_NO_THROW((void)port.autoneg());
  EXPECT_NO_THROW((void)port.pause_params());
}

}  // namespace
