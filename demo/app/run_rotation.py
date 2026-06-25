#!/usr/bin/env python
"""
Standalone script for generating 720-degree rotation videos using DreamX-World-5B.

Usage:
    python demo/app/run_rotation.py --image_path /path/to/image.png
    python demo/app/run_rotation.py --image_path /path/to/image.png --rotation_degrees 360 --duration 12

For help:
    python demo/app/run_rotation.py --help
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
from PIL import Image

# Add project root to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from demo.app.camera_trajectory import (
    generate_smooth_rotation_trajectory,
    get_video_length_for_duration
)
from utils.utils import save_videos_grid


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate 720-degree rotation videos using DreamX-World-5B"
    )
    
    # Input/Output
    parser.add_argument(
        "--image_path", "-i",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./outputs/rotation_demo",
        help="Output directory for generated videos"
    )
    
    # Rotation settings
    parser.add_argument(
        "--rotation_degrees", "-r",
        type=float,
        default=720.0,
        help="Total rotation degrees (default: 720 for 2 full circles)"
    )
    parser.add_argument(
        "--pitch_angle", "-p",
        type=float,
        default=0.0,
        help="Pitch angle in degrees (default: 0)"
    )
    parser.add_argument(
        "--roll_angle",
        type=float,
        default=0.0,
        help="Roll angle in degrees (default: 0)"
    )
    parser.add_argument(
        "--easing",
        type=str,
        default="ease_in_out",
        choices=["linear", "ease_in", "ease_out", "ease_in_out"],
        help="Motion easing type"
    )
    
    # Video settings
    parser.add_argument(
        "--duration",
        type=float,
        default=24.0,
        help="Video duration in seconds (default: 24)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second (default: 24)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Video height (default: 704)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width (default: 1280)"
    )
    
    # Generation settings
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Guidance scale (default: 3.0)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    # Model paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/models/Wan2.2-TI2V-5B",
        help="Path to Wan2.2-TI2V-5B base model"
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="/data/models/DreamX-World-5B-Cam",
        help="Path to DreamX-World-5B-Cam transformer"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/wan2.2/wan_ti2v_5b.yaml",
        help="Path to model config"
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model weight dtype (default: bfloat16)"
    )
    parser.add_argument(
        "--gpu_offload",
        action="store_true",
        default=True,
        help="Enable GPU memory offloading"
    )
    
    # Generation prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default="high quality, cinematic, smooth camera movement, professional photography",
        help="Positive prompt for generation"
    )
    
    return parser.parse_args()


def initialize_pipeline(args):
    """Initialize the DreamX-World pipeline."""
    from omegaconf import OmegaConf
    from diffusers import FlowMatchEulerDiscreteScheduler
    
    from models import AutoencoderKLWan, AutoTokenizer, WanT5EncoderModel
    from models import Wan2_2Transformer3DModel
    from pipeline.pipeline_dreamxworld import Wan2_2_CameraPipeline
    
    print("=" * 60)
    print("Initializing DreamX-World Pipeline")
    print("=" * 60)
    
    config = OmegaConf.load(args.config_path)
    
    if args.weight_dtype == "float16":
        weight_dtype_torch = torch.float16
    elif args.weight_dtype == "bfloat16":
        weight_dtype_torch = torch.bfloat16
    else:
        weight_dtype_torch = torch.float32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)
    
    config['transformer_additional_kwargs']['cam_method'] = 'prope'
    config['transformer_additional_kwargs']['add_control_adapter'] = True
    
    print(f"Loading transformer from: {args.transformer_path}")
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        args.transformer_path,
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        torch_dtype=weight_dtype_torch,
    )
    
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype_torch)
    
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )
    
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype_torch,
    )
    text_encoder = text_encoder.eval()
    
    scheduler = FlowMatchEulerDiscreteScheduler(
        **OmegaConf.to_container(config['scheduler_kwargs'])
    )
    
    pipeline = Wan2_2_CameraPipeline(
        transformer=transformer,
        transformer_2=None,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    
    # Memory optimization
    if args.gpu_offload:
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)
    
    print("Pipeline initialized successfully!")
    print("=" * 60)
    
    return pipeline, device, boundary


def main():
    args = parse_args()
    
    # Validate input
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and validate image
    print(f"Loading image: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    print(f"Image size: {image.size}")
    
    # Initialize pipeline
    pipeline, device, boundary = initialize_pipeline(args)
    
    # Calculate video length
    num_frames = get_video_length_for_duration(
        args.duration,
        fps=args.fps,
        vae_temporal_compression=4
    )
    print(f"Target duration: {args.duration}s, FPS: {args.fps}, Frames: {num_frames}")
    
    # Generate camera trajectory
    print(f"Generating {args.rotation_degrees}° rotation trajectory...")
    camera_condition, _ = generate_smooth_rotation_trajectory(
        num_frames=num_frames,
        rotation_degrees=args.rotation_degrees,
        pitch_angle=args.pitch_angle,
        roll_angle=args.roll_angle,
        radius=0,
        width=args.width,
        height=args.height,
        device=device,
        easing=args.easing
    )
    print("Camera trajectory generated!")
    
    # Generate video
    print("\n" + "=" * 60)
    print("Starting Video Generation")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  - Rotation: {args.rotation_degrees}°")
    print(f"  - Pitch: {args.pitch_angle}°")
    print(f"  - Roll: {args.roll_angle}°")
    print(f"  - Easing: {args.easing}")
    print(f"  - Resolution: {args.width}x{args.height}")
    print(f"  - Duration: {args.duration}s ({num_frames} frames)")
    print(f"  - FPS: {args.fps}")
    print(f"  - Guidance Scale: {args.guidance_scale}")
    print(f"  - Inference Steps: {args.num_inference_steps}")
    print(f"  - Seed: {args.seed}")
    print("=" * 60)
    
    start_time = time.time()
    
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    
    with torch.no_grad():
        sample = pipeline(
            args.prompt,
            num_frames=num_frames,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            start_image=args.image_path,
            control_camera_video=camera_condition,
            boundary=boundary,
            shift=5.0
        ).videos
    
    inference_time = time.time() - start_time
    
    # Save video
    image_name = Path(args.image_path).stem
    output_filename = f"{image_name}_rotation_{int(args.rotation_degrees)}deg_{args.duration}s.mp4"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"\nSaving video to: {output_path}")
    save_videos_grid(sample, output_path, fps=args.fps)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("Video Generation Complete!")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Inference time: {inference_time:.1f}s")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
