#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=1  # 每节点进程数（GPU 数）

# ======================
# Path Configuration
# ======================
MODEL_PATH="/remote-home/share/_hf_models/hfmodel/zd11024/vgllm-qa-vggt-4b"  # [ModelArguments]
GEOMETRY_ENCODER_TYPE="vggt"
GEOMETRY_ENCODER_PATH="facebook/VGGT-1B"
STAGE="vsiData_generation"
CACHE_DIR="./cache"                        # [TrainingArguments]
OTHER_TAG="official_novggt_debug" #"ourckpt_forcevggt" "ourckpt_novggt" "ourckpt_novggt_4090_118"

# ======================
# Model Configuration
# ======================
DATASETS="vsi_20k"
# DATASETS="spar_234k"  # spar_tool_40k llava_hound_tool_10k

# ★ 修正行：补右引号，并把注释放到引号外
OUTPUT_DIR="generatedData/${STAGE}/${DATASETS}/${OTHER_TAG}"  # Directory for saving checkpoints
mkdir -p "$OUTPUT_DIR"

# ======================
# Training Hyperparameters
# ======================
export NCCL_IGNORE_DISABLED_P2P=1

nohup torchrun --nproc_per_node="$NPROC_PER_NODE" \
               --master_addr="$MASTER_ADDR" \
               --master_port="$MASTER_PORT" \
               scripts/data_generation/dataGener_VSI590K.py \
               --model_name_or_path "$MODEL_PATH" \
               --tune_mm_llm True \
               --tune_mm_vision False \
               --tune_mm_mlp False \
               --dataset_use "$DATASETS" \
               --output_dir "$OUTPUT_DIR" \
               --cache_dir "$CACHE_DIR" \
               --bf16 \
               --model_max_length 25600 \
               --data_flatten False \
               --max_pixels $((576*28*28)) \
               --min_pixels $((16*28*28)) \
               --base_interval 2 \
               --video_max_frames 4 \
               --video_min_frames 2 \
               --video_max_frame_pixels $((1664*28*28)) \
               --video_min_frame_pixels $((256*28*28)) \
               --dataloader_num_workers 4 \
               --group_by_modality_length true \
               --seed 0 \
               --report_to "none" \
               --use_geometry_encoder true \
               --geometry_encoder_type "$GEOMETRY_ENCODER_TYPE" \
               --geometry_encoder_path "$GEOMETRY_ENCODER_PATH" \
               --stage "$STAGE" \
               > "${OUTPUT_DIR}/datagenerate.log" 2>&1 &
