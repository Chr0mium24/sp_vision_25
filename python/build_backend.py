from __future__ import annotations

import base64
import csv
import hashlib
import re
import tomllib
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile

PYTHON_ROOT = Path(__file__).resolve().parent
ROOT = PYTHON_ROOT.parent
SRC_ROOT = PYTHON_ROOT / "src"
PACKAGE_NAME = "sp_vision_25_python"
DIST_NAME = "sp_vision_25_python"
WHEEL_TAG = "py3-none-any"


def _read_version() -> str:
    init_py = SRC_ROOT / PACKAGE_NAME / "__init__.py"
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init_py.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError("Unable to determine package version")
    return match.group(1)


VERSION = _read_version()


def _dist_info_dir() -> str:
    return f"{DIST_NAME}-{VERSION}.dist-info"


def _wheel_name() -> str:
    return f"{DIST_NAME}-{VERSION}-{WHEEL_TAG}.whl"


def _iter_package_files() -> Iterable[Path]:
    for path in sorted((SRC_ROOT / PACKAGE_NAME).rglob("*")):
        if path.is_file():
            yield path


def _metadata() -> str:
    return (
        "Metadata-Version: 2.1\n"
        f"Name: {DIST_NAME}\n"
        f"Version: {VERSION}\n"
        "Summary: Python workspace for scripts, tests, and future bindings in sp_vision_25.\n"
    )


def _wheel_file() -> str:
    return (
        "Wheel-Version: 1.0\n"
        "Generator: build_backend\n"
        "Root-Is-Purelib: true\n"
        f"Tag: {WHEEL_TAG}\n"
    )


def _entry_points() -> str:
    return (
        "[console_scripts]\n"
        "sp-vision-diagnose = sp_vision_25_python.diagnose.main:main\n"
        "sp-vision-calibration = sp_vision_25_python.calibration.main:main\n"
    )


def _hash(data: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}", str(len(data))


def _build_wheel(wheel_directory: str) -> str:
    wheel_dir = Path(wheel_directory)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_dir / _wheel_name()

    records: list[tuple[str, str, str]] = []

    with ZipFile(wheel_path, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in _iter_package_files():
            rel_path = file_path.relative_to(SRC_ROOT).as_posix()
            data = file_path.read_bytes()
            zf.writestr(rel_path, data)
            digest, size = _hash(data)
            records.append((rel_path, digest, size))

        metadata_prefix = _dist_info_dir()
        metadata_files = {
            f"{metadata_prefix}/METADATA": _metadata().encode("utf-8"),
            f"{metadata_prefix}/WHEEL": _wheel_file().encode("utf-8"),
            f"{metadata_prefix}/entry_points.txt": _entry_points().encode("utf-8"),
        }
        for rel_path, data in metadata_files.items():
            zf.writestr(rel_path, data)
            digest, size = _hash(data)
            records.append((rel_path, digest, size))

        record_rows = records + [(f"{metadata_prefix}/RECORD", "", "")]
        record_buf = []
        for row in record_rows:
            record_buf.append(",".join(row))
        record_data = "\n".join(record_buf) + "\n"
        zf.writestr(f"{metadata_prefix}/RECORD", record_data.encode("utf-8"))

    return wheel_path.name


def _build_editable_wheel(wheel_directory: str) -> str:
    wheel_dir = Path(wheel_directory)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = wheel_dir / _wheel_name()

    records: list[tuple[str, str, str]] = []
    editable_pth = f"{PACKAGE_NAME}.pth"
    editable_data = f"{SRC_ROOT.as_posix()}\n".encode("utf-8")

    with ZipFile(wheel_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr(editable_pth, editable_data)
        digest, size = _hash(editable_data)
        records.append((editable_pth, digest, size))

        metadata_prefix = _dist_info_dir()
        metadata_files = {
            f"{metadata_prefix}/METADATA": _metadata().encode("utf-8"),
            f"{metadata_prefix}/WHEEL": _wheel_file().encode("utf-8"),
            f"{metadata_prefix}/entry_points.txt": _entry_points().encode("utf-8"),
        }
        for rel_path, data in metadata_files.items():
            zf.writestr(rel_path, data)
            digest, size = _hash(data)
            records.append((rel_path, digest, size))

        record_rows = records + [(f"{metadata_prefix}/RECORD", "", "")]
        record_buf = []
        for row in record_rows:
            record_buf.append(",".join(row))
        record_data = "\n".join(record_buf) + "\n"
        zf.writestr(f"{metadata_prefix}/RECORD", record_data.encode("utf-8"))

    return wheel_path.name


def build_wheel(wheel_directory: str, config_settings=None, metadata_directory=None) -> str:
    return _build_wheel(wheel_directory)


def build_editable(wheel_directory: str, config_settings=None, metadata_directory=None) -> str:
    return _build_editable_wheel(wheel_directory)


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory: str, config_settings=None):
    dist_info = Path(metadata_directory) / _dist_info_dir()
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_metadata(), encoding="utf-8")
    (dist_info / "WHEEL").write_text(_wheel_file(), encoding="utf-8")
    (dist_info / "entry_points.txt").write_text(_entry_points(), encoding="utf-8")
    return dist_info.name


def prepare_metadata_for_build_editable(metadata_directory: str, config_settings=None):
    return prepare_metadata_for_build_wheel(metadata_directory, config_settings=config_settings)
