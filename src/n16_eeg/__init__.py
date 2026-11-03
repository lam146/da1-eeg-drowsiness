"""Core modules for the N16 NeuroSky EEG drowsiness project."""

from .blink import BlinkEvent, detect_blinks
from .signal_features import bandpower_features
from .thinkgear import ThinkGearParser

__all__ = ["BlinkEvent", "ThinkGearParser", "bandpower_features", "detect_blinks"]
__version__ = "0.3.0"
