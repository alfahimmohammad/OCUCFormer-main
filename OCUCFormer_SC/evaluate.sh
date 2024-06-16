MODEL='OCUCFormer' 
BASE_PATH='/media/Data/MRI'
DATASET_TYPE='mrbrain_t1'
MASK_TYPE='cartesian'
ACC_FACTORS='4x'


TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTORS}

PREDICTIONS_PATH='<Set Validation Output Path Here>'

REPORT_PATH='<Set Report Path Here>'

echo python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} --acc-factor ${ACC_FACTORS} --mask-type ${MASK_TYPE} --dataset-type ${DATASET_TYPE}

python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} --acc-factor ${ACC_FACTORS} --mask-type ${MASK_TYPE} --dataset-type ${DATASET_TYPE}
