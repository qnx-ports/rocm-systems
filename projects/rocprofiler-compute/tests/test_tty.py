# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/utils/tty.py."""

import argparse

import pandas as pd

from utils.tty import (
    build_continuation_indent,
    format_args_variant_lines,
    format_node_args,
    format_table_output,
    print_operator_node,
)
from utils.utils_analysis import (
    ARGS_DISPLAY_MAX_VARIANTS,
    CallTreeNode,
    KernelStats,
)


def make_args() -> argparse.Namespace:
    """Minimal args for the plain-table render path."""
    return argparse.Namespace(decimal=2, view=None, normal_unit="per_wave")


def _sample_time_data() -> pd.DataFrame:
    """Metric table mixing a time row, a cycle row, and a count row."""
    return pd.DataFrame({
        "Metric_ID": ["7.2.0", "7.2.1", "7.2.2"],
        "Metric": ["Kernel Time", "Kernel Time (Cycles)", "Non-Time Metric"],
        "Avg": [3446.64, 64499.39, 1000.0],
        "Min": [1769.25, 17269.25, 500.0],
        "Max": [12532.12, 337030.50, 2000.0],
        "Unit": ["ns", "Cycle", "Count"],
    })


def _original_ns_values() -> dict[str, float]:
    """Original nanosecond values for the time row of _sample_time_data."""
    return {"Avg": 3446.64, "Min": 1769.25, "Max": 12532.12}


def test_format_table_output_suppresses_empty_column() -> None:
    """A non-PC-sampling table with an all-'N/A' column is suppressed."""
    df = pd.DataFrame({"Metric": ["a", "b"], "Value": ["N/A", "N/A"]})
    content = format_table_output(
        make_args(),
        {"id": 1101, "title": "Some Table"},
        df,
        "metric_table",
        runs={"only": object()},
    )
    assert content == ""


def test_format_table_output_keeps_pc_sampling_table_21_1() -> None:
    """PC sampling table 21.1 is shown even with an all-'N/A' source column."""
    df = pd.DataFrame({
        "source_line": ["N/A", "N/A"],
        "instruction": ["v_mov", "v_add"],
        "count": [3, 1],
    })
    content = format_table_output(
        make_args(),
        {"id": 2101, "title": "PC Sampling"},
        df,
        "pc_sampling_table",
        runs={"only": object()},
    )
    assert content != ""
    assert "v_mov" in content


def test_format_node_args_present() -> None:
    """A node whose calls all passed the same arguments renders them inline."""
    node = CallTreeNode(
        name="aten::mm",
        args_invocations={"(self=float32[2x2])": {"1@m.py:1", "2@m.py:1"}},
    )
    assert format_node_args(node) == " args=(self=float32[2x2])"


def test_format_node_args_absent() -> None:
    """A node without args (or only empty parens) renders no segment."""
    assert format_node_args(CallTreeNode(name="aten::mm")) == ""
    assert (
        format_node_args(CallTreeNode(name="aten::mm", args_invocations={"()": set()}))
        == ""
    )


def test_format_args_variant_lines_orders_by_call_count() -> None:
    """Variants are listed most frequent first with aligned call counts."""
    node = CallTreeNode(
        name="aten::mm",
        args_invocations={
            "(self=float32[2x2])": {"1@m.py:1"},
            "(self=float32[4x4])": {"2@m.py:1", "3@m.py:1"},
        },
    )
    assert format_args_variant_lines(node) == [
        "args variants:",
        "  2 calls  (self=float32[4x4])",
        "  1 call   (self=float32[2x2])",
    ]


def test_format_args_variant_lines_empty_below_two_variants() -> None:
    """A node with fewer than two variants produces no block."""
    assert format_args_variant_lines(CallTreeNode(name="aten::mm")) == []
    single = CallTreeNode(
        name="aten::mm", args_invocations={"(self=float32[2x2])": {"1@m.py:1"}}
    )
    assert format_args_variant_lines(single) == []


def test_format_args_variant_lines_caps_variant_count() -> None:
    """Variants beyond ARGS_DISPLAY_MAX_VARIANTS collapse into a summary line."""
    node = CallTreeNode(
        name="aten::mm",
        args_invocations={
            f"(self=float32[{size}])": {f"{size}@m.py:1"}
            for size in range(ARGS_DISPLAY_MAX_VARIANTS + 2)
        },
    )
    lines = format_args_variant_lines(node)
    assert len(lines) == ARGS_DISPLAY_MAX_VARIANTS + 2
    assert lines[-1] == "  ... 2 more variants"


def test_format_args_variant_lines_omits_counts_without_context_ids() -> None:
    """Variants with no counted calls are listed without a call-count label."""
    node = CallTreeNode(
        name="aten::mm",
        args_invocations={"(self=float32[2x2])": set(), "(self=float32[4x4])": set()},
    )
    assert format_args_variant_lines(node) == [
        "args variants:",
        "    (self=float32[2x2])",
        "    (self=float32[4x4])",
    ]


def test_print_operator_node_shows_args(capsys) -> None:
    """A single-variant node shows its args inline on the node line."""
    node = CallTreeNode(
        name="aten::mm", args_invocations={"(self=float32[2x2])": {"1@m.py:1"}}
    )
    node.kernels["kernel_gemm"] = KernelStats(launches=1, total_duration_ns=1000.0)

    print_operator_node(node)

    out = capsys.readouterr().out
    assert "aten::mm" in out
    assert "args=(self=float32[2x2])" in out


def test_print_operator_node_shows_args_variants_block(capsys) -> None:
    """A node whose calls passed differing arguments lists each variant."""
    node = CallTreeNode(
        name="aten::mm",
        args_invocations={
            "(self=float32[2x2])": {"1@m.py:1"},
            "(self=float32[4x4])": {"2@m.py:1", "3@m.py:1"},
        },
    )
    node.kernels["kernel_gemm"] = KernelStats(launches=1, total_duration_ns=1000.0)

    print_operator_node(node)

    out = capsys.readouterr().out
    assert "args=" not in out
    assert "args variants:" in out
    assert "2 calls  (self=float32[4x4])" in out
    assert "1 call   (self=float32[2x2])" in out


def test_build_continuation_indent_keeps_pipes() -> None:
    """The branch marker becomes spaces while the parent pipes are kept."""
    assert build_continuation_indent("   |  └─ ", "   |  ") == "   |     "
    assert build_continuation_indent("   |  ", "   |  ") == "   |  "
    assert build_continuation_indent("└─ ", "") == "   "


def make_wrapping_node() -> CallTreeNode:
    """A node whose stats line is long enough to wrap at any terminal width."""
    node = CallTreeNode(name="aten::" + "x" * 60, invocation_ids={"1@m.py:1"})
    node.kernels["kernel_a"] = KernelStats(launches=1, total_duration_ns=1000.0)
    node.kernels["kernel_b"] = KernelStats(launches=1, total_duration_ns=2000.0)
    return node


def test_print_operator_node_keeps_own_pipe_when_node_line_wraps(capsys) -> None:
    """A node with siblings keeps its own connector column on wrapped lines."""
    print_operator_node(make_wrapping_node(), is_last=False, parent_pipes="   |  ")

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("   |  ├─ ")
    assert lines[1].startswith("   |  |  ")


def test_print_operator_node_drops_own_pipe_for_last_child(capsys) -> None:
    """A last child pads its connector column with spaces on wrapped lines."""
    print_operator_node(make_wrapping_node(), is_last=True, parent_pipes="   |  ")

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("   |  └─ ")
    assert lines[1].startswith("   |     ")
