MODEL='OCUCFormer'
DATASET_TYPE='mrbrain_t1'
BASE_PATH='/media/Data/MRI'
MASK_TYPE='cartesian'

ACC_FACTOR='4x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
PREDICTIONS_PATH='<Set Predictions Path Here>'
REPORT_PATH='<Set Report Path Here>'

echo python measures_csv.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} --acc-factor ${ACC_FACTOR}
python measures_csv.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} --acc-factor ${ACC_FACTOR}