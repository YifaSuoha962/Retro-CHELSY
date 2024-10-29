export CUDA_VISIBLE_DEVICES=1,2

tb_port=6018
port=1$tb_port
n_gpu=2

DATASET_NAME=$1
TRAIN_TASK=$2
MODEL_TYPE=$3
LATENT_SZ=$4
BEAM_MODE=$5


torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_port=$port train.py \
--dataset_name="$DATASET_NAME" \
--train_task="$TRAIN_TASK" \
--model_type="$MODEL_TYPE" \
--lat_disc_size=LATENT_SZ \
--beam_module="$BEAM_MODE"