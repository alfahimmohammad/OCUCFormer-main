MODEL='OCUCFormer'
DATASET_TYPE='coronal_pd_h5'
BASE_PATH='/media/Data/MRI/datasets/multicoil/mc_knee'
MASK_TYPE='cartesian'

NC=1
ACC_FACTOR='4x'
TARGET_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/validation/'
PREDICTIONS_PATH='<Predictions Path Here>'
REPORT_PATH='<Report Path Here>'

echo measures_csv.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH}
python measures_csv.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH}

