# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for roofline/benchmark/benchmark_base.py."""

import os
from pathlib import Path
from unittest import mock

import pytest

import utils.utils_profile as utils_profile

# =============================================================================
# GPU Benchmark Locking Tests
# =============================================================================


@pytest.mark.misc
def test_gpu_benchmark_locking(tmp_path, monkeypatch, capsys):
    """Test GPU benchmark locking functions."""
    import fcntl

    import roofline.benchmark.benchmark_base as benchmark_base

    # --- Setup: redirect lock directory to temp path ---
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()

    # Mock GPU UUID
    monkeypatch.setattr(
        benchmark_base.hip,
        "hipGetDeviceProperties",
        lambda d: mock.Mock(uuid=mock.Mock(uuid=bytes([0x01, 0x02, 0x03, 0x04]))),
    )

    # Mock Path to use our temp directory
    original_path = Path

    def mock_path(p):
        if p == "/tmp/rocprof-compute-benchmark":
            return lock_dir
        return original_path(p)

    monkeypatch.setattr(benchmark_base, "Path", mock_path)

    deviceID = 0
    cache_sizes = {}
    # Create Bench_base object in order to call gpu benchmark lock method
    # Device ID list arg doesn't matter since we are just using the base class
    # cache_sizes can be empty for this test since we do not need it to test locking
    testClass = benchmark_base.Bench_base(deviceID, cache_sizes)

    # --- Test lock acquisition and lock file creation ---
    with testClass.gpu_benchmark_lock(deviceID):
        lock_file = lock_dir / "rocprof-compute-benchmark-01020304.lock"
        assert lock_file.exists()

    # --- Test no message when lock acquired immediately ---
    capsys.readouterr()  # Clear previous output
    with testClass.gpu_benchmark_lock(deviceID):
        pass
    output = capsys.readouterr().out
    assert "Waiting" not in output

    # --- Test waiting/acquired messages when lock is contended ---
    call_count = {"count": 0}

    def mock_flock(fd, op):
        call_count["count"] += 1
        if call_count["count"] == 1 and (op & fcntl.LOCK_NB):
            raise BlockingIOError("Lock held by another process")

    monkeypatch.setattr(utils_profile.fcntl, "flock", mock_flock)

    with testClass.gpu_benchmark_lock(deviceID):
        pass

    output = capsys.readouterr().out
    assert "Waiting for GPU 0" in output
    assert "another rocprof-compute benchmark is in progress" in output
    assert "Acquired lock for GPU 0" in output


@pytest.mark.misc
def test_file_lock_creates_world_rw_file(tmp_path):
    """A freshly created lock file must be world-rw (0o666) regardless of umask."""
    import stat

    from utils import utils_profile

    lock_file = tmp_path / "shared.lock"

    # Force a strict umask that would otherwise leave the file owner-only.
    old_umask = os.umask(0o077)
    try:
        with utils_profile.file_lock(lock_file):
            assert lock_file.exists()
    finally:
        os.umask(old_umask)

    file_mode = stat.S_IMODE(os.stat(lock_file).st_mode)
    assert file_mode == 0o666, (
        f"Lock file must be world-rw so any user can acquire it; got "
        f"{oct(file_mode)}. A non-0o666 lock file locks out other users."
    )


@pytest.mark.misc
def test_file_lock_does_not_change_process_umask(tmp_path, monkeypatch):
    """Lock creation must not change process-global umask."""

    def fail_if_called(_mask):
        raise AssertionError("file_lock must not call os.umask()")

    monkeypatch.setattr(utils_profile.os, "umask", fail_if_called)

    with utils_profile.file_lock(tmp_path / "shared.lock"):
        pass


@pytest.mark.misc
def test_file_lock_existing_file_owned_by_other_user(tmp_path, monkeypatch):
    """A lock file owned by another user (no write access) is still lockable."""

    from utils import utils_profile

    lock_file = tmp_path / "shared.lock"
    # Pre-create the lock file (as if another user created it first).
    lock_file.touch()

    real_os_open = os.open
    opened_modes = []

    def fake_os_open(path, flags, *args):
        if flags & os.O_EXCL:
            # Let the create-only attempt fail naturally (file exists).
            return real_os_open(path, flags, *args)
        if flags & os.O_RDWR:
            opened_modes.append("rw")
            raise PermissionError(13, "Permission denied")
        opened_modes.append("ro")
        return real_os_open(path, flags, *args)

    monkeypatch.setattr(utils_profile.os, "open", fake_os_open)

    acquired = False
    with utils_profile.file_lock(lock_file):
        acquired = True

    assert acquired, "Lock must be acquired via read-only fallback"
    assert opened_modes == ["rw", "ro"], (
        "Should attempt read-write first, then fall back to read-only"
    )


@pytest.mark.misc
def test_file_lock_unopenable_file_raises(tmp_path, monkeypatch):
    """If the lock file cannot be opened at all, raise an actionable error."""

    from utils import utils_profile

    lock_file = tmp_path / "shared.lock"
    lock_file.touch()

    real_os_open = os.open

    def fake_os_open(path, flags, *args):
        if flags & os.O_EXCL:
            # Let the create-only attempt fail naturally (file exists).
            return real_os_open(path, flags, *args)
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(utils_profile.os, "open", fake_os_open)

    with pytest.raises(RuntimeError, match="Cannot open lock file"):
        with utils_profile.file_lock(lock_file):
            pass
