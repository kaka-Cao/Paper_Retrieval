#!/bin/bash

SCRIP_DIR=$(echo `cd $(dirname $0); pwd`)
export PATH=/work/cache/env/miniconda3/bin:$PATH


# export TRAIN_DATASET=/work/home/Projects/IntellijenceCustomerService/检索优化_新抽取QA/data/qa_all_samples_1213_neg_count_7.jsonl
export TRAIN_DATASET=

export N_EPOCH=1
export TRAIN_GROUP_SIZE=8  #对比学习参数，可能是1个正样本和7个负样本

export GRADIENT_ACCUMULATION_STEPS=64
export PER_DEVICE_TRAIN_BATCH_SIZE=1
export N_NODES=1
export BATCH_SIZE=`expr ${GRADIENT_ACCUMULATION_STEPS} \* ${PER_DEVICE_TRAIN_BATCH_SIZE} \* ${N_NODES}`

export VERSION=ft_v4sh_bge_large_en_epoch_${N_EPOCH}_bz_${BATCH_SIZE}_trgrp_${TRAIN_GROUP_SIZE}_$(date +"%Y%m%d_%H%M")

# 可选
export WANDB_PROJECT=RAG-From-Scratch-Embedding-Finetune
export WANDB_API_KEY=cee225be654b7d3da070d5ba11d3d6cb4d664221
export WANDB_NAME=${VERSION}

export OUTPUT_DIR=

if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p "${OUTPUT_DIR}"
fi

# model_name_or_path替换为自己的本机路径，或BAAI/bge-large-zh-v1.5
# torchrun --nproc_per_node ${N_NODES} \
python -m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir ${OUTPUT_DIR} \
--model_name_or_path BAAI/bge-large-en-v1.5 \
--train_data ${TRAIN_DATASET} \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs ${N_EPOCH} \
--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 512 \
--train_group_size ${TRAIN_GROUP_SIZE} \
--negatives_cross_device \
--logging_steps 5 \
--save_steps 50 \
--save_total_limit 10 \
--warmup_ratio 0.05 \
--lr_scheduler_type cosine \
--query_instruction_for_retrieval "" 


cp "$SCRIP_DIR/$0" ${OUTPUT_DIR}
