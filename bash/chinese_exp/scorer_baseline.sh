set -e


INPUT_FILE=../../data/mucgec_test/src.txt
OUTPUT_FILE=../../model/baselineScore/baselinemucgec.out
HYP_PARA_FILE=../../model/baselineScore/baselinemucgec.hyp.para
HYP_M2_FILE=../../model/baselineScore/baselinemucgec.hyp.m2.char
REF_M2_FILE=../../model/mucgec.test.m2
SCORE_FILE=../../model/baselineScore/mucgec_baseline.score
# Step1. extract edits from hypothesis file.

paste $INPUT_FILE $OUTPUT_FILE | awk '{print NR"\t"$p}' > $HYP_PARA_FILE  # only for single hypothesis situation

python ../../scorers/ChERRANT/parallel_to_m2.py -f $HYP_PARA_FILE -o $HYP_M2_FILE -g char  # char-level evaluation

# Step2. compare hypothesis edits with reference edits.

python ../../scorers/ChERRANT/compare_m2_for_evaluation.py -hyp $HYP_M2_FILE -ref $REF_M2_FILE > $SCORE_FILE


# Note: you can also extract the reference edits yourself by using parallel_to_m2.py if you have reference sentences.
# You need to process the data into the following format: id \t source \t reference1 \t reference2 \t ... \n

# # word-level evaluation
# HYP_M2_FILE=./samples/demo.hyp.m2.word
# REF_M2_FILE=./samples/demo.ref.m2.word
# python parallel_to_m2.py -f $HYP_PARA_FILE -o $HYP_M2_FILE -g word  
# python compare_m2_for_evaluation.py -hyp $HYP_M2_FILE -ref $REF_M2_FILE