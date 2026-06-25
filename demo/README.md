# DreamX-World 720° Rotation Demo

A demo application that generates 720-degree (2 full circles) rotating scene videos from a single input image using the DreamX-World-5B model with camera trajectory control.

## Features

- **720-degree rotation**: Generate full 2-circle rotation videos around the scene
- **Customizable angles**: Adjust pitch and roll for dynamic viewing angles
- **24-second videos**: Create smooth, cinematic videos at 24 FPS
- **Easing options**: Choose between linear, ease-in, ease-out, and ease-in-out motion curves
- **Interactive Gradio UI**: Easy-to-use web interface for video generation
- **Command-line interface**: Scriptable batch processing support

## Installation

1. Install the main dependencies:

```bash
pip install -r requirements.txt
```

2. Install demo-specific dependencies:

```bash
pip install -r demo/requirements.txt
```

## Quick Start

### Option 1: Command Line

Generate a 720-degree rotation video from an image:

```bash
python demo/app/run_rotation.py \
    --image_path ./demo/034_w.png \
    --rotation_degrees 720 \
    --duration 24 \
    --output_dir ./outputs/rotation_demo
```

Generate a 360-degree rotation with custom parameters:

```bash
python demo/app/run_rotation.py \
    --image_path ./demo/034_w.png \
    --rotation_degrees 360 \
    --pitch_angle 15 \
    --duration 12 \
    --guidance_scale 4.0 \
    --num_inference_steps 50
```

### Option 2: Gradio Web Interface

Launch the interactive web interface:

```bash
python demo/app/demo_app.py
```

Then open `http://localhost:7860` in your browser.

## Usage

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image_path`, `-i` | str | required | Path to input image |
| `--output_dir`, `-o` | str | `./outputs/rotation_demo` | Output directory |
| `--rotation_degrees`, `-r` | float | 720.0 | Total rotation angle |
| `--pitch_angle`, `-p` | float | 0.0 | Up/down tilt (degrees) |
| `--roll_angle` | float | 0.0 | Side tilt (degrees) |
| `--easing` | str | `ease_in_out` | Motion curve type |
| `--duration` | float | 24.0 | Video duration (seconds) |
| `--fps` | int | 24 | Frames per second |
| `--height` | int | 704 | Video height (pixels) |
| `--width` | int | 1280 | Video width (pixels) |
| `--guidance_scale` | float | 3.0 | CFG scale |
| `--num_inference_steps` | int | 50 | Sampling steps |
| `--seed` | int | 42 | Random seed |
| `--model_path` | str | `/data/models/Wan2.2-TI2V-5B` | Base model path |
| `--transformer_path` | str | `/data/models/DreamX-World-5B-Cam` | Transformer path |
| `--prompt` | str | (cinematic) | Generation prompt |

### Python API

```python
from demo.app.camera_trajectory import (
    generate_smooth_rotation_trajectory,
    get_video_length_for_duration
)

# Generate camera trajectory for 720° rotation
num_frames = get_video_length_for_duration(24, fps=24)
camera_condition, _ = generate_smooth_rotation_trajectory(
    num_frames=num_frames,
    rotation_degrees=720.0,
    pitch_angle=0.0,
    roll_angle=0.0,
    easing="ease_in_out"
)

# Use with pipeline...
```

### Gradio Interface

The web interface provides:

1. **Image Upload**: Drag & drop or click to upload an image
2. **Rotation Controls**: Adjust rotation degrees, pitch, and roll
3. **Video Settings**: Set duration, FPS, resolution
4. **Advanced Settings**: Guidance scale, inference steps, seed
5. **Model Configuration**: Set model paths
6. **Generate Button**: Start video generation

## Architecture

```
demo/
├── app/
│   ├── __init__.py           # Package initialization
│   ├── camera_trajectory.py  # Camera trajectory generation
│   ├── demo_app.py           # Gradio web interface
│   └── run_rotation.py       # Command-line script
├── requirements.txt          # Demo dependencies
└── README.md                 # This file
```

### Key Components

1. **camera_trajectory.py**: Generates smooth camera paths with:
   - Yaw (horizontal rotation)
   - Pitch (vertical tilt)
   - Roll (side tilt)
   - Multiple easing curves

2. **demo_app.py**: Gradio interface with:
   - Image upload and preview
   - Real-time parameter adjustment
   - Video preview and download

3. **run_rotation.py**: CLI tool for:
   - Batch processing
   - Scriptable workflows
   - Integration with other tools

## Examples

### Basic 720° Rotation

```bash
python demo/app/run_rotation.py --image_path input.png --rotation_degrees 720
```

### Cinematic Pan with Tilt

```bash
python demo/app/run_rotation.py \
    --image_path input.png \
    --rotation_degrees 360 \
    --pitch_angle 20 \
    --roll_angle 10 \
    --duration 15
```

### Batch Processing

```bash
for img in ./images/*.png; do
    python demo/app/run_rotation.py --image_path "$img" --rotation_degrees 720
done
```

## Tips

- Use **high-quality images** with clear foreground and background separation
- For **smoother motion**, use `ease_in_out` easing
- For **faster rotation**, reduce duration or increase rotation degrees
- **Higher inference steps** produce better quality but take longer
- The **first generation** may be slower due to model warmup
- Use the **same seed** to reproduce results

## Troubleshooting

### Out of Memory Errors

Try:
- Reduce video resolution (height/width)
- Enable `--gpu_offload`
- Use `float16` instead of `bfloat16`

### Poor Quality Output

Try:
- Increase `num_inference_steps`
- Adjust `guidance_scale`
- Use a higher quality input image
- Modify the prompt

### Slow Generation

Try:
- Reduce `num_inference_steps`
- Lower resolution
- Use batch processing with multiple images

## License

Same as the main DreamX-World project.

## Acknowledgments

Built on top of:
- [DreamX-World](https://github.com/your-repo/DreamX-World)
- [Wan2.2](https://github.com/your-repo/wan)
- [Diffusers](https://github.com/huggingface/diffusers)
