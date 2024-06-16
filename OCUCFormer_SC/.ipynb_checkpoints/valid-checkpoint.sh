MODEL='OCUCFormer' 
BASE_PATH='/media/Data/MRI'
DATASET_TYPE='mrbrain_t1'
ACC_FACTORS='4x'
MASK_TYPE='cartesian'
CHECKPOINT='<Set CKPT Path Here>'
OUT_DIR='<Set Output Dir Path Here>'

USMASK_PATH=${BASE_PATH}'/usmasks/'

BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTORS}

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTORS} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH}

python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTORS} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --usmask_path ${USMASK_PATH}