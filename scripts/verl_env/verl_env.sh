conda create --name vgllm_verl --clone vgllm
conda activate vgllm_verl
cd verl
scripts/install_vllm_sglang_mcore.sh
pip install numpy==1.26.4
pip install /remote-home/haohh/_cvpr2025/VG-LLM/scripts/verl_env/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install --no-deps -e .