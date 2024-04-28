MODULE="tinystories.tiny_stories_trainer"
CONFIG_MODULE="tinystories.tiny_stories_config"
CONFIG="fuji-testgpu"
BACKEND="tpu"

COORDINATOR_ADDR="127.0.0.1:5551"
NUM_PROC="1"
PROC_ID="0"

TRAINER_DIR="/tmp/trainer_dir"
DATA_DIR="unused"

rm -rf ${TRAINER_DIR}
mkdir ${TRAINER_DIR}
python -m axlearn.common.launch_trainer_main \
    --module=${MODULE} \
    --config_module=${CONFIG_MODULE} \
    --config=${CONFIG} \
    --trainer_dir=${TRAINER_DIR} \
    --data_dir=${DATA_DIR} \
    --jax_backend=${BACKEND} \
    --distributed_coordinator ${COORDINATOR_ADDR} \
    --num_processes ${NUM_PROC} \
    --process_id ${PROC_ID}
