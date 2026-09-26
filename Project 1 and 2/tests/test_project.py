import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def downloader():
    return load_module("downloader_under_test", "downloader.py")


@pytest.fixture
def sync_dataset():
    return load_module("sync_dataset_under_test", "sync_dataset.py")


def test_download_skips_existing_complete_file(downloader, tmp_path):
    destination = tmp_path / "file.zip"
    destination.write_bytes(b"already complete")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("requests.get should not be called")

    downloader.requests.get = fail_if_called

    assert downloader.download_with_resume("https://example.test/file.zip", destination)
    assert destination.read_bytes() == b"already complete"


def test_download_writes_new_file(downloader, tmp_path, monkeypatch):
    destination = tmp_path / "file.zip"

    class FakeResponse:
        status_code = 200
        headers = {"content-length": "6"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size == 1024 * 1024
            yield b"abc"
            yield b"def"

    monkeypatch.setattr(downloader.requests, "get", lambda *a, **k: FakeResponse())

    assert downloader.download_with_resume("https://example.test/file.zip", destination)
    assert destination.read_bytes() == b"abcdef"
    assert not Path(str(destination) + ".part").exists()


def test_download_resumes_partial_file(downloader, tmp_path, monkeypatch):
    destination = tmp_path / "file.zip"
    partial = Path(str(destination) + ".part")
    partial.write_bytes(b"abc")

    captured = {}

    class FakeResponse:
        status_code = 206
        headers = {"content-length": "3"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"def"

    def fake_get(url, headers, stream, timeout):
        captured.update(headers=headers, stream=stream, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    assert downloader.download_with_resume("https://example.test/file.zip", destination)
    assert captured["headers"] == {"Range": "bytes=3-"}
    assert captured["stream"] is True
    assert captured["timeout"] == 60
    assert destination.read_bytes() == b"abcdef"


def test_download_failure_keeps_partial_file(downloader, tmp_path, monkeypatch):
    destination = tmp_path / "file.zip"
    partial = Path(str(destination) + ".part")
    partial.write_bytes(b"partial")

    def fail(*args, **kwargs):
        raise RuntimeError("network failure")

    monkeypatch.setattr(downloader.requests, "get", fail)

    assert not downloader.download_with_resume("https://example.test/file.zip", destination)
    assert partial.read_bytes() == b"partial"
    assert not destination.exists()


def test_ensure_dataset_files_returns_true_when_all_downloads_succeed(downloader, tmp_path, monkeypatch):
    downloader.DATASET_DIR = tmp_path / "dataset"
    downloader.FILES = {
        "one.zip": {"url": "https://example.test/one.zip", "destination": tmp_path / "dataset" / "one.zip"},
        "two.zip": {"url": "https://example.test/two.zip", "destination": tmp_path / "dataset" / "two.zip"},
    }

    calls = []

    def fake_download(url, destination):
        calls.append((url, destination))
        return True

    monkeypatch.setattr(downloader, "download_with_resume", fake_download)

    assert downloader.ensure_dataset_files()
    assert len(calls) == 2


def test_find_hf_uses_cli_when_available(sync_dataset, monkeypatch):
    class Result:
        returncode = 0

    monkeypatch.setattr(sync_dataset.subprocess, "run", lambda *a, **k: Result())
    assert sync_dataset.find_hf() == "hf"


def test_run_exits_with_subprocess_return_code(sync_dataset, monkeypatch):
    class Result:
        returncode = 7

    monkeypatch.setattr(sync_dataset.subprocess, "run", lambda *a, **k: Result())

    with pytest.raises(SystemExit) as exc:
        sync_dataset.run(["example", "command"])

    assert exc.value.code == 7


def test_sync_main_rejects_unknown_command(sync_dataset, capsys):
    monkeypatch_argv = ["sync_dataset.py", "unknown"]
    original = sys.argv
    try:
        sys.argv = monkeypatch_argv
        sync_dataset.main()
    finally:
        sys.argv = original

    output = capsys.readouterr().out
    assert "Unknown command." in output
    assert "python sync_dataset.py push" in output
    assert "python sync_dataset.py pull" in output
