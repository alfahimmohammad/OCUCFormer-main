MODEL='OCUCFormer'
BASE_PATH='/media/Data/MRI/datasets/multicoil/mc_knee'
DATASET_TYPE='coronal_pd_h5'

NC=1
ACC_FACTOR='4x'

MASK_TYPE='cartesian'
TARGET_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/validation/'
PREDICTIONS_PATH='<Predictions Path Here>'
REPORT_PATH='<Report Path Here>'
echo python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 