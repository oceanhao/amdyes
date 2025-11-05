set -e
# ---------- 基本配置 ----------
PORT=$(( (RANDOM % 20000) + 20000 ))     # 随机端口：20000–39999
export NCCL_NVLS_ENABLE=0
export HF_ENDPOINT="https://hf-mirror.com"
export NCCL_ASYNC_ERROR_HANDLING=1   # 发生通信异常及时报错而不是无限等
export NCCL_BLOCKING_WAIT=1          # collective 出错立刻阻塞报错，便于定位
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IGNORE_DISABLED_P2P=1
benchmark="vsibench"                     # choices: [vsibench, cvbench, blink_spatial,mindcube_full,mmbench_en_dev,videomme]
model_path="/remote-home/haohh/_cvpr2025/VG-LLM/ckpt_saves/mhan/flex-percept-init"
num_processes=4
stage='force_use' #'force_use'、'force_notuse','force_half'(一半vggt)  ("cold_start","stage2-1_rlColdStart"等stage不能在这里使用)
# ---------- 输出与日志 ----------
out_root="logs"                                          # 总日志根目录
out_day="${out_root}/$(TZ='Asia/Shanghai' date +%Y%m%d)" # 日期分桶
mkdir -p "${out_day}"

# 从 model_path 生成可读标签：取“父目录-叶子目录”，并做字符清洗
_parent="$(basename "$(dirname "${model_path}")")"
_leaf="$(basename "${model_path}")"
model_tag="${_parent}-${_leaf}"
# 清洗为文件名友好：替换空格、斜杠、冒号、@ 等
safe_model_tag="$(printf "%s" "${model_tag}" | tr '/:@ ' '____')"

# 生成最终日志名：benchmark + 模型标签 + 时间戳
ts="$(TZ='Asia/Shanghai' date +%Y%m%dT%H%M%S)"
log_file="${out_day}/${benchmark}_${safe_model_tag}_${ts}.log"

# lmms_eval 的结果输出目录（可与日志目录区分）
output_path="${out_day}"

echo "[INFO] benchmark=${benchmark}"
echo "[INFO] model_path=${model_path}"
echo "[INFO] PORT=${PORT}"
echo "[INFO] log_file=${log_file}"
echo "[INFO] output_path=${output_path}"

# ---------- 启动（后台 + nohup + 重定向到日志） ----------
nohup accelerate launch --main_process_port "${PORT}" --num_processes="${num_processes}" -m lmms_eval \
  --model vgllm \
  --model_args "pretrained=${model_path},max_num_frames=12,max_length=25600,stage=${stage}" \
  --tasks "${benchmark}" \
  --batch_size 1 \
  --output_path "${output_path}" \
  > "${log_file}" 2>&1 &

pid=$!
echo "[INFO] started. PID=${pid}"
echo "[INFO] tail -f '${log_file}'"