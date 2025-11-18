set -e

cd /home/jr/research/SeMGEC/data/mucgec_test && cut -f1 MuCGEC_test.txt > src.id

cd /home/jr/research/SeMGEC/data/mucgec_test && cut -f2 MuCGEC_test.txt > src.txt

# File path
TEST_SRC_FILE=../../data/mucgec_test/src.txt

# apply char
if [ ! -f $TEST_SRC_FILE".char" ]; then
  python ../../utils/segment_bert.py <$TEST_SRC_FILE >$TEST_SRC_FILE".char"
fi

# gopar_path=../../model/gopar/emnlp2022_syngec_biaffine-dep-electra-zh-gopar.pt
CoNLL_SUFFIX=conll_predict_gopar
IN_FILE=../../data/mucgec_test/src.txt.char
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
# echo $gopar_path
# CUDA_VISIBLE_DEVICES=0 python ../../src/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path
if [ ! -f $OUT_FILE ]; then
  echo "Parse with GoPar..."
  cd /home/jr/research/SeMGEC/model/gopar && bash preprocess_mucgec_test.sh
fi
# 由于后续使用.txt.CoNLL_SUFFIX等文件，因此将parse的结果文件名复制为.txt.conll_predict_gopar
cp $OUT_FILE "../../data/mucgec_test/src.txt.conll_predict_gopar"
cp $OUT_FILE".probs" "../../data/mucgec_test/src.txt.conll_predict_gopar.probs"

# Subword Align
if [ ! -f $TEST_SRC_FILE".swm" ]; then
  echo "Align subwords and words..."
  python ../../utils/subword_align.py $TEST_SRC_FILE".char" $TEST_SRC_FILE".char" $TEST_SRC_FILE".swm"
fi

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/chinese_mucgec_with_syntax_transformer

WORKER_NUM=4
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_np

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TEST_SRC_FILE $PROCESSED_DIR/test.src
cp $TEST_SRC_FILE".char" $PROCESSED_DIR/test.char.src
cp $TEST_SRC_FILE".swm" $PROCESSED_DIR/test.swm.src

# syntax specific
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CoNLL_SUFFIX conll transformer
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CoNLL_SUFFIX probs transformer

cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/test.conll.src

if [ ! -f $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" ]; then
  echo "Calculate dependency distance..."
  python ../../utils/calculate_dependency_distance.py $TEST_SRC_FILE".${CoNLL_SUFFIX}" $PROCESSED_DIR/test.swm.src $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"
fi

cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/test.dpd.src
cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/test.probs.src

echo "Preprocess..."
mkdir -p $PROCESSED_DIR/bin

python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
       --user-dir ../../src/src_syngec/syngec_model \
       --task syntax-enhanced-translation \
       --only-source \
       --testpref $PROCESSED_DIR/test.char \
       --destdir $PROCESSED_DIR/bin \
       --workers $WORKER_NUM \
       --conll-suffix conll \
       --swm-suffix swm \
       --dpd-suffix dpd \
       --probs-suffix probs \
       --labeldict ../../data/dicts/syntax_label_gec.dict \
       --srcdict ../../data/dicts/chinese_vocab.count.txt \
       --tgtdict ../../data/dicts/chinese_vocab.count.txt

echo "Finished!"
