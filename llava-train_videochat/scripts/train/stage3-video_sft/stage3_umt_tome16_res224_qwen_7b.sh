export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1



DATA_VERSION="data/stage3_short-long_mix_sft.yaml"
DATA_VERSION_CLEAN=$(basename "$DATA_VERSION")

VISION_MODEL_VERSION="umt-large"
VISION_MODEL_VERSION_CLEAN="umt-large"

LLM_VERSION="Your_stage2_checkpoint_path"
LLM_VERSION_CLEAN="Qwen2_7B"

mm_projector_type=tome16_mlp_hd64
PROMPT_VERSION="qwen_2"

MID_RUN_NAME=stage3-${VISION_MODEL_VERSION}-${mm_projector_type}_${LLM_VERSION_CLEAN}_${DATA_VERSION_CLEAN}_$(date +"%Y%m%d_%H%M%S")
echo "MID_RUN_NAME: ${MID_RUN_NAME}"


PARTITION='video'
JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

NUM_GPU=32
# NOTE: If you don't use slurm, please ref to https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/scripts/train/pretrain_clip.sh for training command.
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --ntasks=${NUM_GPU} \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python -u llava/train/train_mem.py \
    --deepspeed scripts/zero1.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_VERSION} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --mm_vision_select_layer -2 \
    --mm_projector_type ${mm_projector_type} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./checkpoints/stage3-video_sft/${MID_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 512 \
    --frames_lowbound 64 \
    --time_msg short \
    --local_num_frames 8 \
    --vision_encode_type video_image \
    --sample_type dynamic_fps1 \
    --mm_local_num_frames 8 \
    --verbose_logging True >> ./output_logs/stage3-video_sft/${MID_RUN_NAME}.log
# You can delete the sdpa attn_implementation if you want to use flash attn