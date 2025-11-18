set -x
# ENGINE=${1:-vllm}
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export USE_OPTIMIZED_MODEL=0        # 与官方一致
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
CFG_path="/remote-home/haohh/_cvpr2025/VG-LLM/verl/verl/trainer/config"
CFG_NAME="ppo_spatialSense_trainer.yaml"
# 路径（按你的环境改）
MODEL_PATH="/remote-home/haohh/_cvpr2025/VG-LLM/ckpt_saves/mhan/flex-percept-init-3e"
REWARD_FILE="/remote-home/haohh/_cvpr2025/VG-LLM/verl/verl/utils/reward_score/spatial_sense_reward.py"


CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 OC_CAUSE=1 python3 -m verl.trainer.main_ppo_spatial \
    --config-path $CFG_path \
    --config-name $CFG_NAME \
    algorithm.adv_estimator=grpo \
    data.max_prompt_length=20480 \
    data.max_response_length=512 \
    data.train_max_samples=32 \
    data.val_max_samples=32 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.external_lib="qwen_vl.model.modeling_qwen2_5_vl" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="$REWARD_FILE" \
    custom_reward_function.name="compute_score" \
    custom_reward_function.reward_kwargs.alpha=1.0 \
    custom_reward_function.reward_kwargs.beta=0.2 \
    custom_reward_function.reward_kwargs.lambda_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_spatial' \
    trainer.experiment_name='qwen2_5_vl_grpo' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=15 \
    trainer.device=cuda $@
