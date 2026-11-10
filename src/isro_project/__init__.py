"""ISRO GNSS C/N0 Prediction Package.

Usage:
    python -m isro_project.01_merge_sessions
    python -m isro_project.02_orientation_model
    python -m isro_project.02_future_model
    python -m isro_project.03_orientation_inference
"""

from .config import ensure_dirs

__all__ = ["ensure_dirs"]
