#!/usr/bin/env python
"""
Test script to verify camera trajectory generation works correctly.
This script doesn't require the model weights.
"""

import sys
import os

# Add project root to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from demo.app.camera_trajectory import (
    generate_smooth_rotation_trajectory,
    generate_720_rotation_trajectory,
    get_video_length_for_duration,
    process_trajectories_to_camera_condition
)
import numpy as np


def test_video_length_calculation():
    """Test video length calculation for 24s at 24fps."""
    print("Testing video length calculation...")
    
    # 24 seconds at 24 fps = 576 frames
    frames = get_video_length_for_duration(24, fps=24)
    print(f"  24s @ 24fps -> {frames} frames")
    assert frames == 577, f"Expected 577 (1+4*144), got {frames}"
    
    # Test with different durations
    frames_10s = get_video_length_for_duration(10, fps=24)
    print(f"  10s @ 24fps -> {frames_10s} frames")
    assert frames_10s >= 240, f"Expected at least 240 frames, got {frames_10s}"
    
    print("  ✓ Video length calculation tests passed!")


def test_720_rotation_trajectory():
    """Test 720-degree rotation trajectory generation."""
    print("\nTesting 720-degree rotation trajectory...")
    
    camera_condition, cam_params = generate_720_rotation_trajectory(
        num_frames=577,
        rotation_degrees=720.0,
        pitch_angle=0.0,
        roll_angle=0.0,
        width=1280,
        height=704,
        device='cpu',
        return_cam_params=True
    )
    
    print(f"  Camera condition keys: {list(camera_condition.keys())}")
    print(f"  viewmats shape: {camera_condition['viewmats'].shape}")
    print(f"  K shape: {camera_condition['K'].shape}")
    
    assert 'viewmats' in camera_condition
    assert 'K' in camera_condition
    assert camera_condition['viewmats'].shape[0] > 0
    assert camera_condition['K'].shape[0] > 0
    
    # Check that we have the right number of latent frames
    latent_frames = camera_condition['viewmats'].shape[0]
    print(f"  Latent frames: {latent_frames}")
    
    print("  ✓ 720° rotation trajectory tests passed!")


def test_easing_options():
    """Test different easing options."""
    print("\nTesting easing options...")
    
    for easing in ['linear', 'ease_in', 'ease_out', 'ease_in_out']:
        camera_condition, _ = generate_smooth_rotation_trajectory(
            num_frames=121,
            rotation_degrees=360.0,
            pitch_angle=0.0,
            roll_angle=0.0,
            width=1280,
            height=704,
            device='cpu',
            easing=easing
        )
        
        print(f"  {easing}: viewmats {camera_condition['viewmats'].shape}")
        assert camera_condition['viewmats'].shape[0] > 0
    
    print("  ✓ Easing options tests passed!")


def test_pitch_and_roll():
    """Test pitch and roll angles."""
    print("\nTesting pitch and roll angles...")
    
    camera_condition, _ = generate_smooth_rotation_trajectory(
        num_frames=121,
        rotation_degrees=180.0,
        pitch_angle=15.0,
        roll_angle=10.0,
        width=1280,
        height=704,
        device='cpu',
        easing='ease_in_out'
    )
    
    print(f"  With pitch=15°, roll=10°:")
    print(f"    viewmats shape: {camera_condition['viewmats'].shape}")
    
    assert camera_condition['viewmats'].shape[0] > 0
    
    print("  ✓ Pitch and roll tests passed!")


def test_rotation_angles():
    """Test various rotation angles."""
    print("\nTesting various rotation angles...")
    
    test_cases = [
        (90.0, "90° rotation"),
        (180.0, "180° rotation"),
        (360.0, "360° rotation (1 circle)"),
        (720.0, "720° rotation (2 circles)"),
        (1080.0, "1080° rotation (3 circles)"),
    ]
    
    for degrees, description in test_cases:
        camera_condition, _ = generate_smooth_rotation_trajectory(
            num_frames=121,
            rotation_degrees=degrees,
            width=1280,
            height=704,
            device='cpu'
        )
        print(f"  {description}: viewmats {camera_condition['viewmats'].shape}")
        assert camera_condition['viewmats'].shape[0] > 0
    
    print("  ✓ Rotation angle tests passed!")


def main():
    print("=" * 60)
    print("Camera Trajectory Generation Tests")
    print("=" * 60)
    
    try:
        test_video_length_calculation()
        test_720_rotation_trajectory()
        test_easing_options()
        test_pitch_and_roll()
        test_rotation_angles()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
