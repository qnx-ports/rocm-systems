# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for mem_chart_gfx9.py - CDNA memory architecture visualization."""

import re

import common
import pytest

from utils import mem_chart_gfx9, mem_chart_gfx11

DEFAULT_TITLE = "3. Memory Chart (Normalization: per_kernel)"

# =============================================================================
# Tests for plot_mem_chart function (gfx9)
# =============================================================================


GFX9_SAMPLE_METRICS = {
    "Wavefront Occupancy": 1,
    "Wave Life": 2,
    "SALU": 3,
    "SMEM": 4,
    "VALU": 5,
    "Matrix Ops": 6,
    "VMEM": 7,
    "LDS": 8,
    "GWS": 9,
    "BR": 10,
    "Active CUs": 11,
    "Num CUs": 12,
    "VGPR": 13,
    "SGPR": 14,
    "LDS Allocation": 15,
    "Scratch Allocation": 16,
    "Wavefronts": 17,
    "Workgroups": 18,
    "LDS Req": 19,
    "LDS Util": 20,
    "LDS Latency": 21,
    "VL1 Rd": 22,
    "VL1 Wr": 23,
    "VL1 Atomic": 24,
    "VL1 Hit": 25,
    "VL1 Lat": 26,
    "VL1 Coalesce": 27,
    "VL1 Stall": 28,
    "sL1D Rd": 29,
    "sL1D Hit": 30,
    "sL1D Lat": 31,
    "IL1 Fetch": 32,
    "IL1 Hit": 33,
    "IL1 Lat": 34,
    "VL1_L2 Rd": 36,
    "VL1_L2 Wr": 37,
    "VL1_L2 Atomic": 38,
    "sL1D_L2 Rd": 39,
    "sL1D_L2 Wr": 40,
    "sL1D_L2 Atomic": 41,
    "IL1_L2 Rd": 42,
    "L2 Hit": 43,
    "L2 Rd": 44,
    "L2 Wr": 45,
    "L2 Atomic": 46,
    "L2 Rd Lat": 47,
    "L2 Wr Lat": 48,
    "Fabric_L2 Rd": 49,
    "Fabric_L2 Wr": 50,
    "Fabric_L2 Atomic": 51,
    "Fabric Rd Lat": 52,
    "Fabric Wr Lat": 53,
    "Fabric Atomic Lat": 54,
    "HBM Rd": 55,
    "HBM Wr": 56,
}


class TestPlotMemChartGfx9:
    """Tests for gfx9 plot_mem_chart - CDNA memory chart generation."""

    def test_returns_non_empty_string(self):
        """Full sample metrics produce a non-empty chart string."""
        result = mem_chart_gfx9.plot_mem_chart(
            dict(GFX9_SAMPLE_METRICS), chart_title=DEFAULT_TITLE
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_metrics(self):
        """Empty metric dict still produces a non-empty chart (N/A placeholders)."""
        result = mem_chart_gfx9.plot_mem_chart({}, chart_title=DEFAULT_TITLE)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_partial_metrics(self):
        """Partial metric dict still produces a non-empty chart."""
        partial = {
            "Wavefront Occupancy": 4,
            "L2 Hit": 75,
            "HBM Rd": 100,
        }
        result = mem_chart_gfx9.plot_mem_chart(partial, chart_title=DEFAULT_TITLE)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_complete_cdna_architecture(self):
        """CDNA output contains every component enabled by the renderer."""
        output = common.strip_ansi(
            mem_chart_gfx9.plot_mem_chart(
                dict(GFX9_SAMPLE_METRICS),
                chart_title=DEFAULT_TITLE,
            )
        )
        expected_components = (
            "Instr Buff",
            "Instr Dispatch",
            "Exec",
            "LDS",
            "Vector L1 Cache",
            "Scalar L1D Cache",
            "Instr L1 Cache",
            "L2 Cache",
            "xGMI/PCIe",
            "Fabric",
            "HBM",
        )

        for component in expected_components:
            assert component in output, f"Missing CDNA component: {component}"

        assert re.search(r"(?<!x)GMI(?!/PCIe)", output)

    def test_gfx9_heading_outside_chart_and_directional_connectors(self):
        """CDNA output prints the heading once outside the chart, with connectors."""
        output = common.strip_ansi(
            mem_chart_gfx9.plot_mem_chart(
                dict(GFX9_SAMPLE_METRICS),
                chart_title=DEFAULT_TITLE,
            )
        )
        output_lines = output.splitlines()

        assert output_lines[0] == DEFAULT_TITLE
        assert "3. Memory Chart" not in "\n".join(output_lines[1:])
        assert "Instr Buff" in output_lines[4]
        assert output.count("(Normalization: per_kernel)") == 1

        for connector_label in ("Req", "Fetch", "Rd", "Wr", "Atomic"):
            assert connector_label in output

        assert re.search(r"<(?!-+>)-{3,}", output)
        assert re.search(r"(?<![<-])-{3,}>", output)
        assert re.search(r"<-{3,}>", output)


@pytest.mark.parametrize(
    ("renderer", "metrics"),
    [
        pytest.param(
            mem_chart_gfx11.plot_mem_chart,
            mem_chart_gfx11.get_sample_metrics(),
            id="gfx115x",
        ),
        pytest.param(
            mem_chart_gfx9.plot_mem_chart,
            dict(GFX9_SAMPLE_METRICS),
            id="gfx9",
        ),
    ],
)
def test_chart_title_appears_as_first_line(renderer, metrics):
    """Both renderers print the caller's chart_title as the first line."""
    chart_title = "7. Memory Chart (Normalization: per_kernel)"
    output = common.strip_ansi(renderer(metrics, chart_title=chart_title))

    assert output.strip().splitlines()[0] == chart_title
    assert "3. Memory Chart" not in output
