#!/bin/bash

####################
# Decoder-Only MoE Training Strategy
# 冻结整个encoder，专注训练decoder中的MoE层
####################

SEED=2022
FAIRSEQ_CLI_PATH=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
MODEL_DIR=/root/autodl-tmp/model/syngec_chinese_bart_moe_decoder_only/
PROCESSED_DIR=/root/autodl-tmp/preprocess/chinese_hsk+lang8_with_syntax_transformer
FAIRSEQ_PATH=../../src/src_syngec/fairseq-0.10.2/fairseq
BART_PATH=/root/autodl-tmp/syngec_chinese_bart_moe_baseline.pt

mkdir -p $MODEL_DIR
mkdir -p $MODEL_DIR/src

cp -r $FAIRSEQ_PATH $MODEL_DIR/src
cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR/src
cp -r ../../src/src_syngec/syngec_model $MODEL_DIR/src
cp ./train_syngec_bart_moe_decoder_only.sh $MODEL_DIR

echo "=== Decoder-Only MoE Training ==="
echo "Freezing: BART parameters + Encoder (sentence + syntax)"
echo "Training: Decoder MoE layers only"


# cos warmupinit=minlr=lr warmupend=maxlr=maxlr w
CUDA_VISIBLE_DEVICES=0 nohup python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR/bin \
    --save-dir $MODEL_DIR \
    --user-dir ../../src/src_syngec/syngec_model \
    --use-syntax \
    --only-gnn \
    --syntax-encoder GCN \
    --use-moe-decoder \
    --moe-num-experts 4 \
    --moe-gate switch \
    --top-k 1 \
    --moe-loss-coef 0.1 \
    --load-balancing-loss-weight 0.02 \
    --expert-dropout 0.15 \
    --moe-gate-warmup-epochs 3 \
    --decoder-moe-only \
    --restore-file $BART_PATH \
    --reset-optimizer \
    --reset-lr-scheduler \
    --reset-dataloader \
    --task syntax-enhanced-translation \
    --arch syntax_enhanced_bart_moe_large \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 4096 \
    --optimizer adam \
    --update-freq 1 \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --max-sentence-length 128 \
    --lr 5e-07 \
    -s src \
    -t tgt \
    --lr-scheduler cosine \
    --max-lr 3e-05 \
    --clip-norm 1.0 \
    --warmup-updates 2000 \
    --criterion label_smoothed_cross_entropy_with_moe \
    --label-smoothing 0.1 \
    --max-epoch 6 \
    --max-update 60000 \
    --share-all-embeddings \
    --adam-betas '(0.9, 0.999)' \
    --log-format tqdm \
    --find-unused-parameters \
    --patience 5 \
    --fp16 \
    --no-epoch-checkpoints \
    --seed $SEED 2>&1 | tee ${MODEL_DIR}/nohup.log &

wait

