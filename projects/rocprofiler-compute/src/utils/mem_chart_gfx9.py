# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CDNA memory chart renderer (MI200/MI300/MI350)."""

from io import StringIO
from typing import Any, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.mem_chart_common import (
    COLORS,
    build_bw_edge_column,
    build_cache_panel,
    build_kernel_panel,
    build_legend,
    colored,
    format_edge,
    format_mem_chart_heading,
    format_value,
    make_arrows,
    mem_chart_cli_main,
    render_chart_to_string,
    scale_or_none,
    strip_ansi,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MEM_CHART_DEFAULT_ROWS: tuple[tuple[str, Union[int, float, None]], ...] = (
    ("Wavefront Occupancy", 8),
    ("Wave Life", 4200),
    ("SALU", 1200),
    ("SMEM", 45),
    ("VALU", 3500),
    ("Matrix Ops", 800),
    ("VMEM", 220),
    ("LDS", 150),
    ("GWS", 0),
    ("BR", 90),
    ("Active CUs (deprecated)", 110),
    ("Num CUs", 110),
    ("VGPR", 64),
    ("SGPR", 32),
    ("LDS Allocation", 32768),
    ("Scratch Allocation", 0),
    ("Wavefronts", 16384),
    ("Workgroups", 256),
    ("Flat Read", 80),
    ("Flat Write", 20),
    ("Flat Atomic", 4),
    ("Buffer Read", 3000),
    ("Buffer Write", 400),
    ("Buffer Atomic", 8),
    ("LDS Req", 150),
    ("LDS Util", 45),
    ("LDS Latency", 28),
    ("LDS Read", None),
    ("LDS Write", None),
    ("LDS Atomic", None),
    ("VL1 Rd", 3200),
    ("VL1 Wr", 480),
    ("VL1 Atomic", 12),
    ("VL1 Hit", 92),
    ("VL1 Lat", 180),
    ("VL1 Coalesce", 87),
    ("VL1 Stall", 5),
    ("VL1_L2 Rd", 256),
    ("VL1_L2 Wr", 48),
    ("VL1_L2 Atomic", 12),
    ("sL1D Rd", 45),
    ("sL1D Hit", 98),
    ("sL1D Lat", 85),
    ("sL1D_L2 Rd", 1),
    ("sL1D_L2 Wr", 0),
    ("sL1D_L2 Atomic", 0),
    ("IL1 Fetch", 32),
    ("IL1 Hit", 99),
    ("IL1 Lat", 42),
    ("IL1_L2 Rd", 1),
    ("L2 Rd", 300),
    ("L2 Wr", 52),
    ("L2 Atomic", 12),
    ("L2 Hit", 85),
    ("L2 Rd Lat", 220),
    ("L2 Wr Lat", 180),
    ("Fabric_L2 Rd", 45),
    ("Fabric_L2 Wr", 8),
    ("Fabric_L2 Atomic", 1),
    ("L2-Fabric Read BW", 45e9),
    ("L2-Fabric Write and Atomic BW", 8e9),
    ("Fabric Rd Lat", 350),
    ("Fabric Wr Lat", 280),
    ("Fabric Atomic Lat", 310),
    ("HBM Rd", 42),
    ("HBM Wr", 7),
    ("HBM Read Traffic", None),
    ("HBM Write and Atomic Traffic", None),
    ("Remote Read Traffic", None),
    ("Remote Write and Atomic Traffic", None),
    ("HBM Read BW", None),
    ("HBM Write BW", None),
    ("HBM Atomic BW", None),
    ("xGMI Read BW", None),
    ("xGMI Write BW", None),
    ("xGMI Atomic BW", None),
    ("PCIe Read BW", None),
    ("PCIe Write BW", None),
    ("PCIe Atomic BW", None),
)

MEM_CHART_PANEL_METRIC_KEYS: tuple[str, ...] = tuple(
    k for k, _ in _MEM_CHART_DEFAULT_ROWS
)

DEFAULT_SAMPLE_METRICS: dict[str, Union[int, float, None]] = dict(
    _MEM_CHART_DEFAULT_ROWS
)


# ---------------------------------------------------------------------------
# Public API helpers
# ---------------------------------------------------------------------------


def normalize_mem_chart_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
    """Filter/reorder input to panel key order; missing keys become None."""
    return {k: metric_dict.get(k) for k in MEM_CHART_PANEL_METRIC_KEYS}


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _extract_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract rendered metrics from the flat dict. Missing keys → None."""
    get = metric_dict.get
    metrics: dict[str, Any] = {}

    # Kernel→L1 request edges
    metrics["flat_read"] = get("Flat Read")
    metrics["flat_write"] = get("Flat Write")
    metrics["flat_atomic"] = get("Flat Atomic")
    metrics["buffer_read"] = get("Buffer Read")
    metrics["buffer_write"] = get("Buffer Write")
    metrics["buffer_atomic"] = get("Buffer Atomic")
    metrics["lds_req"] = get("LDS Req")
    metrics["lds_read"] = get("LDS Read")
    metrics["lds_write"] = get("LDS Write")
    metrics["lds_atomic"] = get("LDS Atomic")
    metrics["smem_rd"] = get("sL1D Rd")
    metrics["icache_rd"] = get("IL1 Fetch")

    # L1 cache panels
    metrics["vl1_hit"] = get("VL1 Hit")
    metrics["sl1d_hit"] = get("sL1D Hit")
    metrics["il1_hit"] = get("IL1 Hit")

    # L1→L2 bytes moved (128B/read, 64B/write, 64B/atomic)
    metrics["vl1_l2_rd_bytes"] = scale_or_none(get("VL1_L2 Rd"), 128)
    metrics["vl1_l2_wr_bytes"] = scale_or_none(get("VL1_L2 Wr"), 64)
    metrics["vl1_l2_atomic_bytes"] = scale_or_none(get("VL1_L2 Atomic"), 64)
    metrics["sl1d_l2_rd_bytes"] = scale_or_none(get("sL1D_L2 Rd"), 64)
    metrics["il1_l2_rd_bytes"] = scale_or_none(get("IL1_L2 Rd"), 64)

    # L2 panel
    metrics["l2_hit"] = get("L2 Hit")

    # L2→Fabric BW (Bytes/s)
    metrics["l2_fabric_read_bw"] = get("L2-Fabric Read BW")
    metrics["l2_fabric_wr_at_bw"] = get("L2-Fabric Write and Atomic BW")

    # Fabric→HBM
    metrics["hbm_rd"] = get("HBM Rd")
    metrics["hbm_wr"] = get("HBM Wr")
    metrics["hbm_read_traffic"] = get("HBM Read Traffic")
    metrics["hbm_wr_at_traffic"] = get("HBM Write and Atomic Traffic")
    metrics["remote_read_traffic"] = get("Remote Read Traffic")
    metrics["remote_wr_at_traffic"] = get("Remote Write and Atomic Traffic")
    metrics["hbm_read_bw"] = get("HBM Read BW")
    metrics["hbm_write_bw"] = get("HBM Write BW")
    metrics["hbm_atomic_bw"] = get("HBM Atomic BW")

    # xGMI / PCIe BW (gfx950 only)
    metrics["xgmi_read_bw"] = get("xGMI Read BW")
    metrics["xgmi_write_bw"] = get("xGMI Write BW")
    metrics["xgmi_atomic_bw"] = get("xGMI Atomic BW")
    metrics["pcie_read_bw"] = get("PCIe Read BW")
    metrics["pcie_write_bw"] = get("PCIe Write BW")
    metrics["pcie_atomic_bw"] = get("PCIe Atomic BW")

    return metrics


# ---------------------------------------------------------------------------
# Diagram building
# ---------------------------------------------------------------------------


_KERNEL_ARROW_LEN = 16
_STD_ARROW_LEN = 12


_VL1D_H = 14
_LDS_H = 10
_SL1D_H = 4
_L1I_H = 4
_TOTAL_H = _VL1D_H + _LDS_H + _SL1D_H + _L1I_H  # 30


def _pad_to(lines: list[str], target: int) -> list[str]:
    """Pad or truncate *lines* to exactly *target* rows (returns new list)."""
    padded = lines + [""] * max(0, target - len(lines))
    return padded[:target]


def _build_kernel_panel() -> Panel:
    """Build the Kernel (shader core) panel at full diagram height."""
    return build_kernel_panel(_TOTAL_H, padding_lines=13)


def _build_request_edges(
    metrics: dict[str, Any],
    arrows: dict[str, str],
) -> Text:
    """Edges from Kernel to L1 caches, aligned to panel heights."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    color_atomic = COLORS["atomic"]
    arrow_left = arrows["left"]
    arrow_right = arrows["right"]
    arrow_both = arrows["both"]

    # VL1D scope — Non-buffer + Buffer requests
    flat_rd = format_edge("Read", metrics["flat_read"])
    flat_wr = format_edge("Write", metrics["flat_write"])
    flat_at = format_edge("Atomic", metrics["flat_atomic"])
    buf_rd = format_edge("Read", metrics["buffer_read"])
    buf_wr = format_edge("Write", metrics["buffer_write"])
    buf_at = format_edge("Atomic", metrics["buffer_atomic"])
    vl1d_lines = [
        "[white]Non-buffer Request[/white]",
        colored(flat_rd, color_read),
        colored(arrow_left, color_read),
        colored(flat_wr, color_write),
        colored(arrow_right, color_write),
        colored(flat_at, color_atomic),
        colored(arrow_both, color_atomic),
        "[white]Buffer Request[/white]",
        colored(buf_rd, color_read),
        colored(arrow_left, color_read),
        colored(buf_wr, color_write),
        colored(arrow_right, color_write),
        colored(buf_at, color_atomic),
        colored(arrow_both, color_atomic),
    ]

    # LDS scope
    if metrics["lds_read"] is not None:
        lds_rd = format_edge("Read", metrics["lds_read"])
        lds_wr = format_edge("Write", metrics["lds_write"])
        lds_at = format_edge("Atomic", metrics["lds_atomic"])
        lds_instr = format_edge("Instr", metrics["lds_req"])
        lds_lines = [
            "[white]LDS[/white]",
            colored(lds_rd, color_read),
            colored(arrow_left, color_read),
            colored(lds_wr, color_write),
            colored(arrow_right, color_write),
            colored(lds_at, color_atomic),
            colored(arrow_both, color_atomic),
            colored(lds_instr, "black"),
            colored(arrow_both, "black"),
        ]
    else:
        lds_lines = [
            "[white]LDS[/white]",
            f"[black]{format_edge('Instr', metrics['lds_req'])}[/black]",
            f"[black]{arrow_both}[/black]",
        ]

    # sL1D scope — SMEM
    sl1d_lines = [
        "[white]SMEM[/white]",
        f"[{color_read}]{format_edge('Read', metrics['smem_rd'])}[/{color_read}]",
        f"[{color_read}]{arrow_left}[/{color_read}]",
    ]

    # L1I scope — ICACHE
    l1i_lines = [
        "[white]ICACHE[/white]",
        f"[{color_read}]{format_edge('Read', metrics['icache_rd'])}[/{color_read}]",
        f"[{color_read}]{arrow_left}[/{color_read}]",
    ]

    lines = (
        _pad_to(vl1d_lines, _VL1D_H)
        + _pad_to(lds_lines, _LDS_H)
        + _pad_to(sl1d_lines, _SL1D_H)
        + _pad_to(l1i_lines, _L1I_H)
    )
    return Text.from_markup("\n".join(lines))


def _build_l1_stack(metrics: dict[str, Any]) -> Table:
    """Build vertically stacked L1 cache panels: VL1D, LDS, sL1D, L1I."""
    color_block = COLORS["block"]

    vl1_panel = build_cache_panel(
        "VL1D",
        [("Hit", metrics["vl1_hit"], "%", COLORS["hit"])],
        width=20,
        height=_VL1D_H,
    )
    lds_panel = Panel(
        "",
        title=f"[bold {color_block}]LDS[/bold {color_block}]",
        border_style=color_block,
        width=20,
        height=_LDS_H,
    )
    sl1d_panel = build_cache_panel(
        "sL1D",
        [("Hit", metrics["sl1d_hit"], "%", COLORS["hit"])],
        width=20,
        height=_SL1D_H,
    )
    l1i_panel = build_cache_panel(
        "L1I",
        [("Hit", metrics["il1_hit"], "%", COLORS["hit"])],
        width=20,
        height=_L1I_H,
    )

    stack = Table.grid(padding=0)
    stack.add_column()
    stack.add_row(vl1_panel)
    stack.add_row(lds_panel)
    stack.add_row(sl1d_panel)
    stack.add_row(l1i_panel)
    return stack


def _build_l1_l2_edges(
    metrics: dict[str, Any],
    arrows: dict[str, str],
) -> Text:
    """L1→L2 edge column: bytes moved (VL1D Rd/Wr/Atomic, sL1D Rd, L1I Rd)."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    color_atomic = COLORS["atomic"]
    arrow_left = arrows["left"]
    arrow_right = arrows["right"]

    vl1_rd_bw = format_value(metrics["vl1_l2_rd_bytes"], "Bytes", 1)
    vl1_wr_bw = format_value(metrics["vl1_l2_wr_bytes"], "Bytes", 1)
    vl1_at_bw = format_value(metrics["vl1_l2_atomic_bytes"], "Bytes", 1)
    sl1d_rd_bw = format_value(metrics["sl1d_l2_rd_bytes"], "Bytes", 1)
    il1_rd_bw = format_value(metrics["il1_l2_rd_bytes"], "Bytes", 1)
    color_sl1d = COLORS["read"]
    color_l1i = COLORS["read"]

    vl1d_lines = [
        "",
        f"[{color_read}]Read[/{color_read}]",
        f"[{color_read}]{vl1_rd_bw}[/{color_read}]",
        f"[{color_read}]{arrow_left}[/{color_read}]",
        "",
        f"[{color_write}]Write[/{color_write}]",
        f"[{color_write}]{vl1_wr_bw}[/{color_write}]",
        f"[{color_write}]{arrow_right}[/{color_write}]",
        "",
        f"[{color_atomic}]Atomic[/{color_atomic}]",
        f"[{color_atomic}]{vl1_at_bw}[/{color_atomic}]",
        f"[{color_atomic}]{arrows['both']}[/{color_atomic}]",
    ]
    sl1d_lines = [
        f"[{color_sl1d}]Read[/{color_sl1d}]",
        f"[{color_sl1d}]{sl1d_rd_bw}[/{color_sl1d}]",
        f"[{COLORS['read']}]{arrow_left}[/{COLORS['read']}]",
    ]
    l1i_lines = [
        f"[{color_l1i}]Read[/{color_l1i}]",
        f"[{color_l1i}]{il1_rd_bw}[/{color_l1i}]",
        f"[{COLORS['read']}]{arrow_left}[/{COLORS['read']}]",
    ]

    lines = (
        _pad_to(vl1d_lines, _VL1D_H)
        + _pad_to([], _LDS_H)
        + _pad_to(sl1d_lines, _SL1D_H)
        + _pad_to(l1i_lines, _L1I_H)
    )
    return Text.from_markup("\n".join(lines))


def _build_l2_panel(metrics: dict[str, Any]) -> Panel:
    return build_cache_panel(
        "L2",
        [("Hit", metrics["l2_hit"], "%", COLORS["hit"])],
        width=18,
        height=_TOTAL_H,
    )


def _build_l2_fabric_edges(
    metrics: dict[str, Any],
    arrows: dict[str, str],
) -> Text:
    """L2→Fabric edges: Read BW and Write/Atomic BW."""
    return build_bw_edge_column(
        [
            (
                "Read BW",
                format_value(metrics["l2_fabric_read_bw"], "Bytes/s", 1),
                "left",
                COLORS["read"],
            ),
            (
                "Write/Atomic BW",
                format_value(metrics["l2_fabric_wr_at_bw"], "Bytes/s", 1),
                "right",
                COLORS["write"],
            ),
        ],
        arrows,
        height=_TOTAL_H,
        center=True,
    )


def _ip_block(
    title: str,
    width: int,
    border_style: str = COLORS["block"],
    content: str = "",
) -> Panel:
    """Create an IP block panel with standard height."""
    return Panel(
        content,
        title=f"[bold {border_style}]{title}[/bold {border_style}]",
        border_style=border_style,
        width=width,
        height=_TOTAL_H,
    )


def _build_fabric_content(metrics: dict[str, Any]) -> str:
    """Build Rich markup for the Data Fabric panel (gfx908–gfx942)."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    arrows = make_arrows(8)
    read_edge = format_edge("Read", metrics["hbm_rd"])
    write_edge = format_edge("Write", metrics["hbm_wr"])
    hbm_rd_pct = format_value(metrics["hbm_read_traffic"], "%")
    hbm_wr_pct = format_value(metrics["hbm_wr_at_traffic"], "%")
    remote_rd_pct = format_value(metrics["remote_read_traffic"], "%")
    remote_wr_pct = format_value(metrics["remote_wr_at_traffic"], "%")
    lines = [
        "[white]To/From HBM[/white]",
        colored(read_edge, color_read),
        colored(arrows["left"], color_read),
        colored(write_edge, color_write),
        colored(arrows["right"], color_write),
        "",
        f"[white]HBM   Rd {hbm_rd_pct}[/white]",
        f"[white]      Wr {hbm_wr_pct}[/white]",
        f"[white]Remote Rd {remote_rd_pct}[/white]",
        f"[white]       Wr {remote_wr_pct}[/white]",
    ]
    return "\n".join(lines)


def _build_hbm_content(
    metrics: dict[str, Any],
) -> str:
    """Build Rich markup for the HBM panel (gfx950 BW metrics)."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    color_atomic = COLORS["atomic"]
    color_rd = color_read
    color_wr = color_write
    color_at = color_atomic
    rd_bw = format_value(metrics["hbm_read_bw"], "Bytes/s", 1)
    wr_bw = format_value(metrics["hbm_write_bw"], "Bytes/s", 1)
    at_bw = format_value(metrics["hbm_atomic_bw"], "Bytes/s", 1)
    lines = [
        f"[{color_rd}]Read BW[/{color_rd}]",
        f"[{color_rd}]{rd_bw}[/{color_rd}]",
        "",
        f"[{color_wr}]Write BW[/{color_wr}]",
        f"[{color_wr}]{wr_bw}[/{color_wr}]",
        "",
        f"[{color_at}]Atomic BW[/{color_at}]",
        f"[{color_at}]{at_bw}[/{color_at}]",
    ]
    return "\n".join(lines)


def _build_xgmi_row(console: Console, metrics: dict[str, Any], fabric_col: int) -> None:
    """Render the xGMI block above the main diagram with BW metrics."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    color_atomic = COLORS["atomic"]
    read_bw = format_value(metrics.get("xgmi_read_bw"), "Bytes/s", 1)
    write_bw = format_value(metrics.get("xgmi_write_bw"), "Bytes/s", 1)
    atomic_bw = format_value(metrics.get("xgmi_atomic_bw"), "Bytes/s", 1)

    xgmi_panel = Panel(
        "[dim]XGMI (to Peer GPU)[/dim]",
        border_style=COLORS["block"],
        width=24,
        height=3,
    )
    xgmi_layout = Table.grid(padding=0)
    xgmi_layout.add_column(width=fabric_col)
    xgmi_layout.add_column()
    xgmi_layout.add_row("", xgmi_panel)
    console.print(xgmi_layout)

    pad = " " * (fabric_col + 3)
    arrow_lines = Text.from_markup(
        f"{pad}[{color_read}]|^  Read BW    {read_bw}[/{color_read}]\n"
        f"{pad}[{color_write}]||  Write BW   {write_bw}[/{color_write}]\n"
        f"{pad}[{color_atomic}]||  Atomic BW  {atomic_bw}[/{color_atomic}]"
    )
    console.print(arrow_lines)


def _build_pcie_row(console: Console, metrics: dict[str, Any], fabric_col: int) -> None:
    """Render the PCIe block below the main diagram with BW metrics."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    color_atomic = COLORS["atomic"]
    read_bw = format_value(metrics.get("pcie_read_bw"), "Bytes/s", 1)
    write_bw = format_value(metrics.get("pcie_write_bw"), "Bytes/s", 1)
    atomic_bw = format_value(metrics.get("pcie_atomic_bw"), "Bytes/s", 1)

    pcie_width = 46
    pad = " " * (fabric_col + 3)
    arrow_lines = Text.from_markup(
        f"{pad}[{color_read}]||  Read BW    {read_bw}[/{color_read}]\n"
        f"{pad}[{color_write}]||  Write BW   {write_bw}[/{color_write}]\n"
        f"{pad}[{color_atomic}]V|  Atomic BW  {atomic_bw}[/{color_atomic}]"
    )
    console.print(arrow_lines)

    pcie_panel = Panel(
        "[dim]PCIe (to CPU or Non-XGMI connected GPU)[/dim]",
        border_style=COLORS["block"],
        width=pcie_width,
        height=3,
    )
    pcie_layout = Table.grid(padding=0)
    pcie_layout.add_column(width=fabric_col - (pcie_width - 24) // 2)
    pcie_layout.add_column()
    pcie_layout.add_row("", pcie_panel)
    console.print(pcie_layout)


def _measure_grid(grid: Table, console_width: int = 240) -> tuple[int, int]:
    """Return (total_width, fabric_col) by rendering the grid to a buffer."""
    buf = StringIO()
    tmp = Console(file=buf, force_terminal=False, width=console_width, height=5)
    tmp.print(grid)
    clean = strip_ansi(buf.getvalue())
    total = 0
    fabric_col = 0
    for line in clean.split("\n"):
        stripped = line.rstrip()
        total = max(total, len(stripped))
        if "Data Fabric" in stripped and not fabric_col:
            fabric_col = stripped.index("Data Fabric") - 5
    return total, fabric_col


def _print_scope_bar(console: Console, total_width: int, fabric_col: int) -> None:
    """Print scope bar spanning *total_width*, split at *fabric_col*."""
    gpu_label = " [dim]GPU (XCD)[/dim] "
    fabric_label = " [dim]Fabric / Memory[/dim] "
    gpu_label_plain = " GPU (XCD) "
    fabric_label_plain = " Fabric / Memory "

    gpu_section = fabric_col - 1  # -1 for leading '|'
    fab_section = total_width - fabric_col - 2  # -2 for middle '|' and trailing '|'

    gpu_pad_l = (gpu_section - len(gpu_label_plain)) // 2
    gpu_pad_r = gpu_section - len(gpu_label_plain) - gpu_pad_l
    fab_pad_l = (fab_section - len(fabric_label_plain)) // 2
    fab_pad_r = fab_section - len(fabric_label_plain) - fab_pad_l

    console.print(
        f"|{'-' * gpu_pad_l}{gpu_label}{'-' * gpu_pad_r}"
        f"|{'-' * fab_pad_l}{fabric_label}{'-' * fab_pad_r}|"
    )


# ---------------------------------------------------------------------------
# Main diagram assembly
# ---------------------------------------------------------------------------


def create_mem_chart_diagram(
    metric_dict: dict[str, Any],
    console: Console,
    show_debug: bool = False,
    chart_title: str = "",
    gpu_arch: Optional[str] = None,
) -> None:
    """Create the CDNA memory diagram matching the reference PNG layout."""
    metrics = _extract_metrics(metric_dict)
    kernel_arrows = make_arrows(_KERNEL_ARROW_LEN)
    std_arrows = make_arrows(_STD_ARROW_LEN)
    is_gfx950 = gpu_arch is not None and gpu_arch.startswith("gfx950")

    # Build main diagram grid first (needed to measure width for scope bar)
    kernel = _build_kernel_panel()
    req_edges = _build_request_edges(metrics, kernel_arrows)
    l1_stack = _build_l1_stack(metrics)
    l1_l2_edges = _build_l1_l2_edges(metrics, std_arrows)
    l2 = _build_l2_panel(metrics)
    l2_fab_edges = _build_l2_fabric_edges(metrics, std_arrows)
    if is_gfx950:
        fabric = _ip_block("Data Fabric", 22, COLORS["block"])
        hbm_content = _build_hbm_content(metrics)
        hbm = _ip_block("HBM", 18, COLORS["block"], hbm_content)
    else:
        fabric_content = _build_fabric_content(metrics)
        fabric = _ip_block("Data Fabric", 22, COLORS["block"], fabric_content)
        hbm = _ip_block("HBM", 10, COLORS["block"])
    mall = _ip_block("MALL", 18, COLORS["block"])
    umc = _ip_block("UMC", 8, COLORS["block"])

    main_layout = Table.grid(padding=0)
    for _ in range(10):
        main_layout.add_column()
    main_layout.add_row(
        kernel,
        req_edges,
        l1_stack,
        l1_l2_edges,
        l2,
        l2_fab_edges,
        fabric,
        mall,
        umc,
        hbm,
    )

    chart_width, fabric_col = _measure_grid(main_layout, console.width)

    console.print()
    if chart_title:
        console.print(f"[bold]{chart_title}[/bold]")

    if is_gfx950:
        _build_xgmi_row(console, metrics, fabric_col)
        console.print()
    _print_scope_bar(console, chart_width, fabric_col)
    console.print()

    console.print(main_layout)
    console.print()

    if is_gfx950:
        _build_pcie_row(console, metrics, fabric_col)
        console.print()

    console.print(build_legend())
    console.print()

    if show_debug:
        console.print("[dim]Architecture Notes (CDNA):[/dim]")
        console.print("  VL1D: Per-CU vector data cache (Buffer/Non-buffer requests)")
        console.print("  LDS: Local Data Share, on-CU scratchpad")
        console.print("  sL1D: Per-CU scalar data cache (SMEM requests)")
        console.print("  L1I: Per-CU instruction cache (ICACHE requests)")
        console.print("  L2 (TCC): Shared last-level cache")
        console.print("  Data Fabric: Infinity Fabric interconnect")
        console.print("  MALL: Mid-level Address Lookup Layer (MI300+)")
        console.print("  UMC: Unified Memory Controller")
        console.print("  HBM: High Bandwidth Memory")
        console.print(
            "  xGMI: Inter-GPU link (MI350 has individual counters;"
            " earlier cards use 'traffic to remote')"
        )
        console.print(
            "  PCIe: Host/non-xGMI link (MI350 has individual counters;"
            " earlier cards use 'traffic to remote')"
        )
        console.print()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_mem_chart(
    normal_unit: str,
    metric_dict: dict[str, Any],
    *,
    chart_title: Optional[str] = None,
    gpu_arch: Optional[str] = None,
) -> str:
    """Render the CDNA memory chart and return as a string."""
    resolved_heading = (
        format_mem_chart_heading(normal_unit, panel_id=300)
        if chart_title is None
        else chart_title
    )
    kwargs: dict[str, Any] = {"chart_title": resolved_heading}
    if gpu_arch is not None:
        kwargs["gpu_arch"] = gpu_arch
    return render_chart_to_string(
        create_mem_chart_diagram,
        metric_dict,
        normalize_mem_chart_metrics,
        console_width=240,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mem_chart_cli_main(
        "CDNA Memory Chart - CLI",
        create_mem_chart_diagram,
        normalize_mem_chart_metrics,
        DEFAULT_SAMPLE_METRICS,
        console_width=240,
    )


if __name__ == "__main__":
    main()
