"""TimesFM forecasting pipeline for the Jane Street Real-Time Market Data Forecasting data."""

# On Windows, importing pyarrow.dataset after torch can trigger a native access
# violation in some environments. This is the one place that preloads it; every
# submodule imports through this package, so it always runs before torch.
import pyarrow.dataset  # noqa: F401

__version__ = "0.1.0"
