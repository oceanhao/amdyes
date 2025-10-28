set -e
export LMMS_EVAL_LAUNCHER="accelerate"

export NCCL_NVLS_ENABLE=0
benchmark=cvbench # choices: [vsibench, cvbench, blink_spatial]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")
model_path=/remote-home/share/_hf_models/hfmodel/Qwen/Qwen2.5-VL-7B-Instruct

accelerate launch --num_processes=2 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32 \
    --tasks ${benchmark} \
    --batch_size 1 \
    --output_path $output_path