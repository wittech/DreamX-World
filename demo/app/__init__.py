"""
Demo package for DreamX-World 720-degree rotation.

Usage:
    from demo.app import run_rotation
    
    # Or run the Gradio interface:
    python demo/app/demo_app.py
"""

from .camera_trajectory import (
    generate_720_rotation_trajectory,
    generate_smooth_rotation_trajectory,
    get_video_length_for_duration,
)

__all__ = [
    "generate_720_rotation_trajectory",
    "generate_smooth_rotation_trajectory", 
    "get_video_length_for_duration",
]
