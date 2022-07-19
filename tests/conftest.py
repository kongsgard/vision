import tempfile
from pathlib import Path

import pytest

import vision
from vision.image import Image

KEEP_FILES = False


@pytest.fixture
def test_directory() -> Path:
    """Return the path to the tests/ directory"""
    return Path(__file__).parent.resolve()


@pytest.fixture
def output_directory(tmp_path: Path) -> Path:
    """Directory to store exported files"""
    if KEEP_FILES:
        return Path(__file__).parent.resolve() / "sample_files"
    else:
        return tmp_path


@pytest.fixture
def plane_image(test_directory: Path) -> Image:
    """Return a loaded image of a plane"""
    return vision.image.load(
        test_directory / "sample_files" / "plane_640x480.bin", shape=(480, 640)
    )
