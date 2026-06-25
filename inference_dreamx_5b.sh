# ======================== Model Path ========================
MODEL_NAME="/data/models/Wan2.2-TI2V-5B"    # Path to the folder containing the Wan2.2-5B-TI2V model weights.
CONFIG_PATH="./configs/wan2.2/wan_ti2v_5b.yaml" # Path to model config file.
TRANSFORMER_PATH="/data/models/DreamX-World-5B-Cam"  # Path to the folder containing the DreamX model weights.
# ====================== Basic settings ======================
INPUT_DIR="./configs/dreamx/eval.json"          # Json file of inputs, containing image, prompt, and camera control.
OUTPUT_DIR="./outputs/"          # Directory of saving output video.
SAMPLE_HEIGHT=704                # Height of the input image/output video.
SAMPLE_WIDTH=1280                # Width of the input image/output video.
VIDEO_LENGTH=121                 # Number of frames (must satisfy 1+4k pattern, e.g., 81, 121).
FPS=24                           # FPS of the output video.
GUIDANCE_SCALE=3.0               # CFG scale.
NUM_INFERENCE_STEPS=50           # Number of sampling steps.
SEED=42                          # Random seed for noise sampling.

# ====================== Camera Control ======================
CAM_METHOD="prope"               # camera control method
ADD_CONTROL_ADAPTER="--add_control_adapter"

# ======================== Multi-GPU ========================
WEIGHT_DTYPE="bfloat16"          # inference dtype.
ULYSSES_DEGREE=8                 # ulysses degree, 1 for no ulysses.
RING_DEGREE=1                    # ring degree, 1 for no ring.
CUDA_DEVICES="0,1,2,3,4,5,6,7"   # Specify GPUs, e.g., "4,5,6,7". Empty = use all available.

# ======================== Build Command ========================
if [ -n "${CUDA_DEVICES}" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
    echo "Using GPUs: ${CUDA_DEVICES}"
fi

TOTAL_GPUS=$((ULYSSES_DEGREE * RING_DEGREE))
if [ "${TOTAL_GPUS}" -gt 1 ]; then
    CMD="torchrun --nproc_per_node=${TOTAL_GPUS} --master_addr=localhost --master_port=12345 inference_dreamx5b.py"
else
    CMD="python inference_dreamx5b.py"
fi

CMD="${CMD} --config_path ${CONFIG_PATH}"
CMD="${CMD} --model_name ${MODEL_NAME}"
CMD="${CMD} --transformer_path ${TRANSFORMER_PATH}"
CMD="${CMD} --input_dir ${INPUT_DIR}"
CMD="${CMD} --output_dir ${OUTPUT_DIR}"
CMD="${CMD} --cam_method ${CAM_METHOD}"
CMD="${CMD} ${ADD_CONTROL_ADAPTER}"
CMD="${CMD} --sample_size ${SAMPLE_HEIGHT} ${SAMPLE_WIDTH}"
CMD="${CMD} --video_length ${VIDEO_LENGTH}"
CMD="${CMD} --fps ${FPS}"
CMD="${CMD} --guidance_scale ${GUIDANCE_SCALE}"
CMD="${CMD} --num_inference_steps ${NUM_INFERENCE_STEPS}"
CMD="${CMD} --seed ${SEED}"
CMD="${CMD} --weight_dtype ${WEIGHT_DTYPE}"
CMD="${CMD} --ulysses_degree ${ULYSSES_DEGREE}"
CMD="${CMD} --ring_degree ${RING_DEGREE}"

echo "=============================================="
echo "  Inference"
echo "=============================================="
echo "Command: ${CMD}"
echo "=============================================="

eval ${CMD}
