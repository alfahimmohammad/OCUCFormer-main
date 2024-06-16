MODEL='OCUCFormer'
BASE_PATH='/media/Data/MRI'

DATASET_TYPE='mrbrain_t1'
MASK_TYPE='cartesian'
ACC_FACTORS='4x'
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
EXP_DIR='<Set Path Here>'
TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/usmasks/'
CHECKPOINT=${EXP_DIR}'/model.pt'

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE}

python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE} 

# --resume --checkpoint ${CHECKPOINT}
