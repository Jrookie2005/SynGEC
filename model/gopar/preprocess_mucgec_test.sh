gopar_path=emnlp2022_syngec_biaffine-dep-electra-zh-gopar.pt
CoNLL_SUFFIX=conll_predict_gopar
IN_FILE=../../data/mucgec_test/src.txt.char
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
echo $gopar_path
CUDA_VISIBLE_DEVICES=0 python ../../src/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path