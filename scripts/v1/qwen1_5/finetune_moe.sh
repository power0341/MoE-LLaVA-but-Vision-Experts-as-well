#!/bin/bash

moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.01
JSON_FOLDER="/hotdata/xly/llava-moe-data/train_json"
IMAGE_FOLDER="/hotdata/xly/llava-moe-data"
# cd ~/MoE-LLaVA
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --master_port 60001 moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules mlp.w1 mlp.w2 mlp.c_proj wg \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path /data1/xly/models/llava_moe/llavaqwen1.5-1.8b-finetune/checkpoint-72 \
    --version qwen \
    --data_path ${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/nlp_tune.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower "vision_experts_group" \
    --vision_experts_group_hidden_size 2304 \
    --vision_expert_clip_image_tower "google/siglip-so400m-patch14-384" \
    --vision_expert_clip_mm_vision_select_layer -2 \
    --vision_expert_depth_anything_image_tower "LiheYoung/depth-anything-small-hf" \
    --vision_expert_owlv2_image_tower "google/owlv2-base-patch16-ensemble" \
    --image_projector_type mlp2x_gelu_norm \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data1/xly/models/llava_moe/llavaqwen1.5-1.8b-finetune-moe \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "/data1/xly/models"
