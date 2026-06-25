"""
Gradio web interface for DreamX-World 720-degree rotation demo.
"""

import os
import sys
import io
import time
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import torch

# Add project root to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DreamXInferenceRunner:
    """Handles model loading and inference for DreamX-World."""
    
    def __init__(self):
        self.pipeline = None
        self.device = None
        self.config = None
        self.vae = None
        self.boundary = None
        self.initialized = False
        
    def initialize(
        self,
        model_path: str,
        transformer_path: str,
        config_path: str = "./configs/wan2.2/wan_ti2v_5b.yaml",
        weight_dtype: str = "bfloat16",
        GPU_memory_mode: str = "model_cpu_offload"
    ):
        """Initialize the model pipeline."""
        from omegaconf import OmegaConf
        from diffusers import FlowMatchEulerDiscreteScheduler
        
        from dist import set_multi_gpus_devices
        from models import (AutoencoderKLWan, AutoTokenizer, WanT5EncoderModel)
        from models import Wan2_2Transformer3DModel
        from pipeline.pipeline_dreamxworld import Wan2_2_CameraPipeline
        from utils.fm_solvers import FlowDPMSolverMultistepScheduler
        
        config = OmegaConf.load(config_path)
        self.config = config
        
        if weight_dtype == "float16":
            weight_dtype_torch = torch.float16
        elif weight_dtype == "bfloat16":
            weight_dtype_torch = torch.bfloat16
        else:
            weight_dtype_torch = torch.float32
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)
        self.boundary = boundary
        
        config['transformer_additional_kwargs']['cam_method'] = 'prope'
        config['transformer_additional_kwargs']['add_control_adapter'] = True
        
        print(f"Loading transformer from: {transformer_path}")
        transformer = Wan2_2Transformer3DModel.from_pretrained(
            transformer_path,
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            torch_dtype=weight_dtype_torch,
        )
        
        Chosen_AutoencoderKL = AutoencoderKLWan
        vae = Chosen_AutoencoderKL.from_pretrained(
            os.path.join(model_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        ).to(weight_dtype_torch)
        self.vae = vae
        
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )
        
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype_torch,
        )
        text_encoder = text_encoder.eval()
        
        scheduler = FlowMatchEulerDiscreteScheduler(
            **OmegaConf.to_container(config['scheduler_kwargs'])
        )
        
        self.pipeline = Wan2_2_CameraPipeline(
            transformer=transformer,
            transformer_2=None,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        )
        
        if GPU_memory_mode == "sequential_cpu_offload":
            self.pipeline.enable_sequential_cpu_offload(device=self.device)
        elif GPU_memory_mode == "model_cpu_offload":
            self.pipeline.enable_model_cpu_offload(device=self.device)
        else:
            self.pipeline.to(device=self.device)
        
        self.initialized = True
        print("Model initialized successfully!")
        
    def generate_video(
        self,
        image: Image.Image,
        rotation_degrees: float = 720.0,
        pitch_angle: float = 0.0,
        roll_angle: float = 0.0,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 577,
        fps: int = 24,
        guidance_scale: float = 3.0,
        num_inference_steps: int = 50,
        seed: int = 42,
        prompt: str = "",
        negative_prompt: str = "",
        easing: str = "ease_in_out"
    ) -> Tuple[str, str]:
        """
        Generate a rotating video from the input image.
        
        Returns:
            Tuple of (video_path, info_message)
        """
        from demo.app.camera_trajectory import (
            generate_smooth_rotation_trajectory,
            get_video_length_for_duration
        )
        
        if not self.initialized:
            return None, "Error: Model not initialized"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_image_path = os.path.join(temp_dir, "input_image.png")
            image.save(temp_image_path)
            
            actual_video_length = get_video_length_for_duration(
                num_frames / fps,
                fps=fps,
                vae_temporal_compression=4
            )
            
            camera_condition, _ = generate_smooth_rotation_trajectory(
                num_frames=actual_video_length,
                rotation_degrees=rotation_degrees,
                pitch_angle=pitch_angle,
                roll_angle=roll_angle,
                radius=0,
                width=width,
                height=height,
                device=self.device,
                easing=easing
            )
            
            if not prompt:
                prompt = "high quality, cinematic, smooth camera movement"
            
            if not negative_prompt:
                negative_prompt = (
                    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
                    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
                    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
                    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
                )
            
            generator = torch.Generator(device="cpu").manual_seed(seed)
            
            start_time = time.time()
            
            with torch.no_grad():
                sample = self.pipeline(
                    prompt,
                    num_frames=actual_video_length,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    generator=generator,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    start_image=temp_image_path,
                    control_camera_video=camera_condition,
                    boundary=self.boundary,
                    shift=5.0
                ).videos
            
            inference_time = time.time() - start_time
            
            output_video_path = os.path.join(temp_dir, "output.mp4")
            
            from utils.utils import save_videos_grid
            save_videos_grid(sample, output_video_path, fps=fps)
            
            final_output_path = os.path.join(temp_dir, f"rotation_{int(rotation_degrees)}deg.mp4")
            import shutil
            shutil.copy(output_video_path, final_output_path)
            
            info_message = (
                f"Generated successfully!\n"
                f"Duration: {inference_time:.1f}s\n"
                f"Rotation: {rotation_degrees}°\n"
                f"Frames: {actual_video_length}\n"
                f"FPS: {fps}"
            )
            
            return final_output_path, info_message


def create_demo_interface():
    """Create the Gradio demo interface."""
    
    runner = DreamXInferenceRunner()
    
    with gr.Blocks(
        title="DreamX-World 720° Rotation Demo",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
        )
    ) as demo:
        gr.Markdown(
            """
            # DreamX-World 720° Rotation Demo
            
            Upload an image and generate a **720-degree rotating scene video** using DreamX-World-5B model.
            
            ### Features:
            - **720-degree rotation**: Full 2-circle rotation around the scene
            - **Customizable angles**: Adjust pitch and roll for dynamic views
            - **24-second video**: Generate smooth, cinematic videos
            - **Easing options**: Choose smooth or linear motion curves
            
            ### Tips:
            - Upload a clear, high-quality image for best results
            - Images with distinct foreground and background work best
            - The model will generate smooth camera movement around the scene
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input Image")
                image_input = gr.Image(
                    type="pil",
                    label="Upload your image",
                    height=300
                )
                
                gr.Markdown("### ⚙️ Rotation Settings")
                
                rotation_degrees = gr.Slider(
                    minimum=0,
                    maximum=3600,
                    value=720,
                    step=30,
                    label="Rotation Degrees",
                    info="Total rotation angle (720 = 2 full circles)"
                )
                
                pitch_angle = gr.Slider(
                    minimum=-45,
                    maximum=45,
                    value=0,
                    step=5,
                    label="Pitch Angle",
                    info="Up/down tilt in degrees"
                )
                
                roll_angle = gr.Slider(
                    minimum=-45,
                    maximum=45,
                    value=0,
                    step=5,
                    label="Roll Angle",
                    info="Side tilt in degrees"
                )
                
                easing = gr.Dropdown(
                    choices=["linear", "ease_in", "ease_out", "ease_in_out"],
                    value="ease_in_out",
                    label="Motion Easing"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎬 Output Video")
                video_output = gr.Video(
                    label="Generated Video",
                    height=400
                )
                info_output = gr.Textbox(
                    label="Generation Info",
                    lines=4,
                    interactive=False
                )
        
        with gr.Row():
            gr.Markdown("### 🎚️ Advanced Settings")
        
        with gr.Row():
            with gr.Column(scale=1):
                height = gr.Slider(
                    minimum=480,
                    maximum=960,
                    value=704,
                    step=32,
                    label="Height",
                    info="Video height in pixels"
                )
            with gr.Column(scale=1):
                width = gr.Slider(
                    minimum=640,
                    maximum=1920,
                    value=1280,
                    step=32,
                    label="Width",
                    info="Video width in pixels"
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                duration_seconds = gr.Slider(
                    minimum=5,
                    maximum=30,
                    value=24,
                    step=1,
                    label="Duration (seconds)",
                    info="Video duration"
                )
                fps = gr.Slider(
                    minimum=16,
                    maximum=30,
                    value=24,
                    step=1,
                    label="FPS",
                    info="Frames per second"
                )
            with gr.Column(scale=1):
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=3.0,
                    step=0.5,
                    label="Guidance Scale",
                    info="Higher = more adherence to prompt"
                )
                num_inference_steps = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Inference Steps",
                    info="More steps = better quality but slower"
                )
        
        with gr.Row():
            seed = gr.Number(
                value=42,
                label="Random Seed",
                info="Same seed = reproducible results"
            )
            prompt = gr.Textbox(
                label="Custom Prompt (optional)",
                placeholder="Leave empty for default prompt...",
                lines=2
            )
        
        def calculate_frames(duration, fps):
            return int(duration * fps)
        
        num_frames_display = gr.Number(
            value=calculate_frames(24, 24),
            label="Total Frames",
            interactive=False
        )
        
        gr.Markdown(
            """
            ### 📋 Model Configuration
            
            Please configure the model paths before generating:
            """
        )
        
        with gr.Row():
            model_path = gr.Textbox(
                label="Base Model Path",
                value="/data/models/Wan2.2-TI2V-5B",
                info="Path to Wan2.2-TI2V-5B model"
            )
            transformer_path = gr.Textbox(
                label="Transformer Path",
                value="/data/models/DreamX-World-5B-Cam",
                info="Path to DreamX-World-5B-Cam transformer"
            )
        
        init_button = gr.Button("🚀 Initialize Model", variant="primary")
        init_status = gr.Textbox(label="Initialization Status", lines=2, interactive=False)
        
        generate_button = gr.Button("🎬 Generate Video", variant="primary", size="lg")
        
        init_button.click(
            fn=lambda m, t: (
                runner.initialize(m, t),
                "Model initialized successfully!" if runner.initialized else "Initialization failed"
            ),
            inputs=[model_path, transformer_path],
            outputs=init_status
        )
        
        duration_seconds.change(
            fn=calculate_frames,
            inputs=[duration_seconds, fps],
            outputs=num_frames_display
        )
        fps.change(
            fn=calculate_frames,
            inputs=[duration_seconds, fps],
            outputs=num_frames_display
        )
        
        def generate_video_wrapper(
            image,
            rotation_degrees,
            pitch_angle,
            roll_angle,
            height,
            width,
            duration_seconds,
            fps,
            guidance_scale,
            num_inference_steps,
            seed,
            prompt,
            negative_prompt,
            easing
        ):
            if image is None:
                return None, "Please upload an image first!"
            
            if not runner.initialized:
                return None, "Please initialize the model first!"
            
            num_frames = int(duration_seconds * fps)
            
            try:
                video_path, info = runner.generate_video(
                    image=image,
                    rotation_degrees=rotation_degrees,
                    pitch_angle=pitch_angle,
                    roll_angle=roll_angle,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    fps=fps,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    easing=easing
                )
                return video_path, info
            except Exception as e:
                return None, f"Error: {str(e)}"
        
        generate_button.click(
            fn=generate_video_wrapper,
            inputs=[
                image_input, rotation_degrees, pitch_angle, roll_angle,
                height, width, duration_seconds, fps,
                guidance_scale, num_inference_steps, seed,
                prompt, gr.Textbox(value="", visible=False),  # negative_prompt hidden for simplicity
                easing
            ],
            outputs=[video_output, info_output]
        )
        
        gr.Markdown(
            """
            ### 📖 Usage Guide
            
            1. **Upload Image**: Click on the image upload area and select an image
            2. **Configure Rotation**: Adjust rotation degrees, pitch, and roll angles
            3. **Set Video Parameters**: Choose height, width, duration, and FPS
            4. **Initialize Model**: Click "Initialize Model" (do this once)
            5. **Generate**: Click "Generate Video" to create your rotating scene
            
            ### ⚠️ Notes
            
            - First generation may take longer due to model warmup
            - Higher inference steps produce better quality but are slower
            - The video will be generated in the temporary output directory
            """
        )
    
    return demo


def main():
    """Main entry point for the demo."""
    demo = create_demo_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
