echo "=== Decoder-Only Training Complete ==="
echo "Model saved to: $MODEL_DIR"
echo "Log file: $MODEL_DIR/nohup.log"

#!/bin/bash

####################
# Decoder-Only MoE Training Strategy
# 冻结整个encoder，专注训练decoder中的MoE层
####################

SEED=2022
FAIRSEQ_CLI_PATH=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
MODEL_DIR=../../model/syngec_chinese_bart_moe_decoder_only/
PROCESSED_DIR=../../preprocess/chinese_hsk+lang8_with_syntax_transformer
FAIRSEQ_PATH=../../src/src_syngec/fairseq-0.10.2/fairseq
BART_PATH=../../model/syngec/syngec_chinese_bart_moe_baseline.pt

mkdir -p $MODEL_DIR
mkdir -p $MODEL_DIR/src

cp -r $FAIRSEQ_PATH $MODEL_DIR/src
cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR/src
cp -r ../../src/src_syngec/syngec_model $MODEL_DIR/src
cp ./train_syngec_bart_moe_decoder_only.sh $MODEL_DIR

echo "=== Decoder-Only MoE Training ==="
echo "Freezing: BART parameters + Encoder (sentence + syntax)"
echo "Training: Decoder MoE layers only"

CUDA_VISIBLE_DEVICES=0 nohup python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR/bin \
    --save-dir $MODEL_DIR \
    --user-dir ../../src/src_syngec/syngec_model \
    --use-syntax \
    --only-gnn \
    --syntax-encoder GCN \
    --use-moe-decoder \
    --moe-num-experts 4 \
    --moe-gate switch \
    --top-k 2 \
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
    --max-tokens 16 \
    --optimizer adam \
    --update-freq 16 \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --max-sentence-length 128 \
    --lr 5e-04 \
    --warmup-updates 10 \
    -s src \
    -t tgt \
    --lr-scheduler polynomial_decay \
    --clip-norm 0.5 \
    --criterion label_smoothed_cross_entropy_with_moe \
    --label-smoothing 0.1 \
    --max-epoch 1 \
    --share-all-embeddings \
    --adam-betas '(0.9, 0.999)' \
    --log-format tqdm \
    --find-unused-parameters \
    --keep-last-epochs 1 \
    --patience 1 \
    --validate-interval 1 \
    --save-interval 1 \
    --log-interval 10 \
    --fp16-no-flatten-grads \
    --seed $SEED >${MODEL_DIR}/nohup.log 2>&1 &

wait

echo "=== Decoder-Only Training Complete ==="
echo "Model saved to: $MODEL_DIR"
echo "Log file: $MODEL_DIR/nohup.log"