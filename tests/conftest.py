"""Pytest fixtures for bodypipe tests."""

import sys
from pathlib import Path

import pytest

# Ensure bodypipe root and GVHMR root are importable
BODYPIPE_ROOT = Path(__file__).resolve().parent.parent
GVHMR_ROOT = BODYPIPE_ROOT.parent / "GVHMR"

if str(BODYPIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(BODYPIPE_ROOT))
if str(GVHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GVHMR_ROOT))


@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication instance for the entire test session."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(["--platform", "offscreen"])
    yield app


@pytest.fixture
def session():
    """Create a fresh Session instance."""
    from models.session import Session

    return Session()


@pytest.fixture
def pipeline_config():
    """Create a fresh PipelineConfig instance."""
    from models.pipeline_config import PipelineConfig

    return PipelineConfig()
