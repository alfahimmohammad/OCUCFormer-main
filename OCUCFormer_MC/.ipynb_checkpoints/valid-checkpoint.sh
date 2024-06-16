MODEL='OCUCFormer'
BASE_PATH='/media/Data/MRI/datasets/multicoil/mc_knee'
DATASET_TYPE='coronal_pd_h5'
#<<ACC_FACTOR_4x
BATCH_SIZE=1
NC=1
DEVICE='cuda:0'
ACC_FACTOR='4x'
MASK_TYPE='cartesian'
CHECKPOINT='<CKPT Path Here>'
OUT_DIR='<Results Path Here>'
DATA_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/validation/'

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --nc ${NC}

python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --nc ${NC}
