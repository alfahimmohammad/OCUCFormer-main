MODEL='OCUCFormer'
BASE_PATH='/media/Data/MRI/datasets/multicoil/mc_knee'
DATASET_TYPE='coronal_pd_h5'

#<<ACC_FACTOR_4x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='5x'
NC=1
MASK_TYPE='cartesian'
# EXP_DIR='<Set Path Here>'
EXP_DIR='/media/data16TB/MRI/fahim_experiments/OCUCFormer_MC/exp/ocucrn/coronal_pd_h5_4x_cartesian/OCUCFormer/'
TRAIN_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/train/'
VALIDATION_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/validation/'

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --nc ${NC}

python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --nc ${NC} --model ${MODEL}

