#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=4  # Automatically detects available GPUs

# ======================
# Path Configuration
# ======================
# train sr一定要从含特殊vggt token的ckpt开始！！！！！！！获取带vggt token的方法：/remote-home/haohh/_cvpr2025/VG-LLM/scripts/add_spatialToken.py
MODEL_PATH="/remote-home/haohh/_cvpr2025/VG-LLM/ckpt_saves/mhan/flex-percept-init-3e"  # [ModelArguments] Pretrained model path
stage="cold_start" #[cold_start, cold_startv2]
GEOMETRY_ENCODER_TYPE="vggt"
GEOMETRY_ENCODER_PATH="facebook/VGGT-1B"
out_root="train_output"                 # Directory for saving checkpoints
CACHE_DIR="./cache"                        # [TrainingArguments] Cache directory for models
other_path_tag="_from_init-3e"
OUTPUT_DIR="${out_root}/${stage}${other_path_tag}"
mkdir -p $OUTPUT_DIR

# ======================
# Model Configuration
# ======================
DATASETS="spar_234k,llava_hound_64k"   
# DATASETS="spar_234k"                 # [DataArguments] Dataset with sampling rate

# ======================
# Training Hyperparameters
# ======================
LR=3e-6
total_batch_size=64
GRADIENT_ACCUMULATION_STEPS=$(($total_batch_size / $NPROC_PER_NODE))
# GRADIENT_ACCUMULATION_STEPS=4
export NCCL_IGNORE_DISABLED_P2P=1
nohup torchrun --nproc_per_node=$NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            src/qwen_vl/train/train_qwen.py \
            --model_name_or_path $MODEL_PATH \
            --tune_mm_llm True \
            --tune_mm_vision False \
            --tune_mm_mlp False \
            --dataset_use $DATASETS \
            --output_dir $OUTPUT_DIR \
            --cache_dir $CACHE_DIR \
            --bf16 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
            --learning_rate $LR \
            --mm_projector_lr 1e-5 \
            --vision_tower_lr 0 \
            --optim adamw_torch \
            --model_max_length 25600 \
            --data_flatten False \
            --base_interval 2 \
            --video_max_frames 8 \
            --video_min_frames 4 \
            --num_train_epochs 1 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 50 \
            --save_steps 1000 \
            --save_total_limit 4 \
            --deepspeed "scripts/zero2_opt.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 4 \
            --group_by_modality_length true \
            --seed 42 \
            --report_to "none" \
            --use_geometry_encoder true \
            --geometry_encoder_type $GEOMETRY_ENCODER_TYPE \
            --geometry_encoder_path $GEOMETRY_ENCODER_PATH \
            --stage $stage \
            > ${OUTPUT_DIR}/train_sr.log 2>&1 &
