"""
Camera trajectory generator for 720-degree rotation demo.
Generates camera pose sequences for rotating around the scene.
"""

import numpy as np
import torch
import math
from typing import Optional, List, Tuple


class Camera:
    """Camera class for storing camera parameters."""
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def get_relative_pose(cam_params):
    """Calculate camera poses relative to the first frame's pose."""
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    
    target_cam_c2w = abs_c2ws[0]
    abs2rel = np.linalg.inv(target_cam_c2w)
    
    ret_poses = [abs2rel @ abs_c2w for abs_c2w in abs_c2ws]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def _invert_SE3(transforms):
    """Invert a batch of 4x4 SE(3) matrices."""
    assert transforms.shape[-2:] == (4, 4)
    rotation_inv = transforms[..., :3, :3].transpose(-1, -2)
    result = torch.zeros_like(transforms)
    result[..., :3, :3] = rotation_inv
    result[..., :3, 3] = -torch.einsum('...ij,...j->...i', rotation_inv, transforms[..., :3, 3])
    result[..., 3, 3] = 1.0
    return result


def generate_720_rotation_trajectory(
    num_frames: int,
    rotation_degrees: float = 720.0,
    pitch_angle: float = 0.0,
    roll_angle: float = 0.0,
    radius: float = 0.0,
    width: int = 1280,
    height: int = 704,
    fx: float = 0.8,
    fy: float = 0.8,
    cx: float = 0.5,
    cy: float = 0.5,
    device: str = 'cpu',
    return_cam_params: bool = False
) -> Tuple[dict, List]:
    """
    Generate a 720-degree (or any degree) rotation camera trajectory.
    
    Args:
        num_frames: Total number of frames to generate
        rotation_degrees: Total rotation degrees (default 720 for 2 full rotations)
        pitch_angle: Additional pitch angle in degrees (positive = look up)
        roll_angle: Additional roll angle in degrees
        radius: Orbit radius (0 = stationary rotation, positive = circular orbit)
        width: Target video width
        height: Target video height
        fx, fy, cx, cy: Camera intrinsic parameters
        device: Device for tensor operations
        return_cam_params: Whether to return raw camera parameters
    
    Returns:
        camera_condition: Dict with 'viewmats' and 'K' tensors for model input
        cam_params: Optional raw camera parameters list
    """
    trajectories = []
    current_position = np.zeros(3)
    current_rotation = np.eye(3)
    
    rotation_radians = np.radians(rotation_degrees)
    pitch_radians = np.radians(pitch_angle)
    roll_radians = np.radians(roll_angle)
    
    rotation_per_frame = rotation_radians / num_frames if num_frames > 0 else 0
    
    # Create rotation matrix for initial pitch and roll
    if pitch_angle != 0 or roll_angle != 0:
        cos_p, sin_p = np.cos(pitch_radians), np.sin(pitch_radians)
        cos_r, sin_r = np.cos(roll_radians), np.sin(roll_radians)
        cos_y, sin_y = 1, 0  # No initial yaw
        
        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        R_initial = np.array([
            [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
            [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
            [-sin_p, cos_p * sin_r, cos_p * cos_r]
        ])
        current_rotation = R_initial
    
    for frame_idx in range(num_frames):
        yaw_radians = frame_idx * rotation_per_frame
        
        if frame_idx > 0:
            cos_y, sin_y = np.cos(yaw_radians), np.sin(yaw_radians)
            cos_p, sin_p = np.sin(pitch_radians), -np.cos(pitch_radians) if pitch_angle != 0 else 0
            cos_p, sin_p = np.cos(pitch_radians), np.sin(pitch_radians)
            cos_r, sin_r = np.cos(roll_radians), np.sin(roll_radians)
            
            R_yaw = np.array([
                [cos_y, 0, sin_y],
                [0, 1, 0],
                [-sin_y, 0, cos_y]
            ])
            R_pitch = np.array([
                [1, 0, 0],
                [0, cos_p, -sin_p],
                [0, sin_p, cos_p]
            ])
            R_roll = np.array([
                [cos_r, -sin_r, 0],
                [sin_r, cos_r, 0],
                [0, 0, 1]
            ])
            
            current_rotation = R_yaw @ R_pitch @ R_roll
        
        if radius > 0:
            orbit_x = radius * np.sin(yaw_radians)
            orbit_z = radius * np.cos(yaw_radians) - radius
            current_position = np.array([orbit_x, 0, orbit_z])
        
        c2w_rotation = current_rotation
        c2w_translation = current_position
        
        w2c_rotation = c2w_rotation.T
        w2c_translation = -w2c_rotation @ c2w_translation
        w2c_mat = np.hstack((w2c_rotation, w2c_translation.reshape(3, 1)))
        
        frame_params = [frame_idx, fx, fy, cx, cy, 0, 0] + list(w2c_mat.flatten())
        trajectories.append(frame_params)
    
    return process_trajectories_to_camera_condition(
        trajectories, width, height, device, return_cam_params
    )


def generate_smooth_rotation_trajectory(
    num_frames: int,
    rotation_degrees: float = 720.0,
    pitch_angle: float = 0.0,
    roll_angle: float = 0.0,
    radius: float = 0.0,
    width: int = 1280,
    height: int = 704,
    fx: float = 0.8,
    fy: float = 0.8,
    cx: float = 0.5,
    cy: float = 0.5,
    device: str = 'cpu',
    return_cam_params: bool = False,
    easing: str = 'ease_in_out'
) -> Tuple[dict, List]:
    """
    Generate a smooth rotation trajectory with easing.
    
    Args:
        easing: 'linear', 'ease_in', 'ease_out', 'ease_in_out'
    """
    def ease_function(t):
        if easing == 'linear':
            return t
        elif easing == 'ease_in':
            return t * t
        elif easing == 'ease_out':
            return t * (2 - t)
        elif easing == 'ease_in_out':
            return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t
        return t
    
    trajectories = []
    current_position = np.zeros(3)
    current_rotation = np.eye(3)
    
    rotation_radians = np.radians(rotation_degrees)
    pitch_radians = np.radians(pitch_angle)
    roll_radians = np.radians(roll_angle)
    
    pitch_r, _ = np.sin(pitch_radians), np.cos(pitch_radians)
    roll_r, _ = np.sin(roll_radians), np.cos(roll_radians)
    
    for frame_idx in range(num_frames):
        t = frame_idx / (num_frames - 1) if num_frames > 1 else 0
        eased_t = ease_function(t)
        yaw_radians = eased_t * rotation_radians
        
        cos_y, sin_y = np.cos(yaw_radians), np.sin(yaw_radians)
        cos_p, sin_p = np.cos(pitch_radians), np.sin(pitch_radians)
        cos_r, sin_r = np.cos(roll_radians), np.sin(roll_radians)
        
        R_yaw = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        R_pitch = np.array([
            [1, 0, 0],
            [0, cos_p, -sin_p],
            [0, sin_p, cos_p]
        ])
        R_roll = np.array([
            [cos_r, -sin_r, 0],
            [sin_r, cos_r, 0],
            [0, 0, 1]
        ])
        
        current_rotation = R_yaw @ R_pitch @ R_roll
        
        if radius > 0:
            eased_radius_t = ease_function(t) if easing == 'linear' else ease_function(t)
            orbit_x = radius * np.sin(eased_t * rotation_radians)
            orbit_z = radius * np.cos(eased_t * rotation_radians) - radius
            current_position = np.array([orbit_x, 0, orbit_z])
        
        c2w_rotation = current_rotation
        c2w_translation = current_position
        
        w2c_rotation = c2w_rotation.T
        w2c_translation = -w2c_rotation @ c2w_translation
        w2c_mat = np.hstack((w2c_rotation, w2c_translation.reshape(3, 1)))
        
        frame_params = [frame_idx, fx, fy, cx, cy, 0, 0] + list(w2c_mat.flatten())
        trajectories.append(frame_params)
    
    return process_trajectories_to_camera_condition(
        trajectories, width, height, device, return_cam_params
    )


def process_trajectories_to_camera_condition(
    trajectories: List,
    width: int,
    height: int,
    device: str = 'cpu',
    return_cam_params: bool = False
) -> Tuple[dict, List]:
    """
    Process trajectory parameters into camera condition dict for model input.
    """
    cam_params = [Camera(traj) for traj in trajectories]
    
    # Get relative poses
    c2w_poses = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2w_poses, dtype=torch.float32, device=device)
    
    # Align to VAE temporal downsampling (1+4k pattern)
    n_frames = len(cam_params)
    latent_frame_count = 1 + (n_frames - 1) // 4
    
    src_indices = np.arange(n_frames, dtype=np.float64)
    tgt_indices = np.linspace(0, n_frames - 1, latent_frame_count)
    
    from utils.pose_utils import interpolate_camera_poses
    cam_params_aligned = interpolate_camera_poses(cam_params, src_indices, tgt_indices)
    
    c2w_poses_aligned = get_relative_pose(cam_params_aligned)
    c2ws_aligned = torch.as_tensor(c2w_poses_aligned, dtype=torch.float32, device=device)
    
    # Compute viewmats (w2c)
    viewmats = _invert_SE3(c2ws_aligned)
    
    # Compute intrinsics
    default_intrinsic = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ]
    fx_norm = default_intrinsic[0][0] / (default_intrinsic[0][2] * 2)
    fy_norm = default_intrinsic[1][1] / (default_intrinsic[1][2] * 2)
    
    T_latent = viewmats.shape[0]
    Ks = torch.zeros((T_latent, 3, 3), device=device, dtype=torch.float32)
    Ks[:, 0, 0] = fx_norm
    Ks[:, 1, 1] = fy_norm
    Ks[:, 0, 2] = 0
    Ks[:, 1, 2] = 0
    Ks[:, 2, 2] = 1.0
    
    camera_condition = {
        'viewmats': viewmats,
        'K': Ks,
    }
    
    if return_cam_params:
        return camera_condition, trajectories
    return camera_condition, None


def get_video_length_for_duration(
    target_duration_seconds: float,
    fps: int = 24,
    vae_temporal_compression: int = 4
) -> int:
    """
    Calculate video length for target duration.
    
    Args:
        target_duration_seconds: Target video duration in seconds
        fps: Frames per second
        vae_temporal_compression: VAE temporal compression ratio (4 for Wan2.2)
    
    Returns:
        video_length: Number of frames for the model
    """
    total_frames = int(target_duration_seconds * fps)
    
    # Align to 1 + 4k pattern (required by VAE)
    if total_frames <= 1:
        return 1
    video_length = ((total_frames - 1) // vae_temporal_compression) * vae_temporal_compression + 1
    
    return video_length
