# Editable paths
CONDA_HOME="/shared/thangakr/conda"
CONDA_ENV_NAME="tot"
OUTPUT_DIR="/shared_new/thangakr/axlearn_out"
NEURON_DUMP_PATH=${PWD}/neuron_dump
HLO_DUMP_PATH=${PWD}/hlo_dump

# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --distribution-strategy=llm-training"
#export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--verify-hlo --num-concat-graphs=8'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"

# Neuron PJRT flags
export NEURON_WHILE_LOOP_UNROLL=1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1

# Neuron runtime flags
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1

# Source conda environment
# source ${CONDA_HOME}/bin/activate ${CONDA_ENV_NAME}

flags=${XLA_FLAGS:-""}
#export XLA_FLAGS="$flags --xla_force_host_platform_device_count=32 --xla_disable_hlo_passes=while-loop-all-reduce-code-motion,neuron_all_gather_combiner,flip_reduce_scatter_transpose,neuron_move_all_gather_while_loop,neuron_all_gather_duplicate_remover,gather_simplifier,scatter_simplifier"
#export XLA_FLAGS="$flags --xla_force_host_platform_device_count=32 --xla_disable_hlo_passes=neuron_move_all_gather_while_loop"
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
#export TF_CPP_MIN_LOG_LEVEL=0
#export TF_CPP_MAX_VLOG_LEVEL=2

export NEURON_RT_ROOT_COMM_ID="127.0.0.1:5552"
export NEURON_PJRT_PROCESSES_NUM_DEVICES="32"
export NEURON_PJRT_PROCESS_INDEX=0
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_LOG_LEVEL="warn"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1

MODULE="tinystories.tiny_stories_trainer"
CONFIG_MODULE="tinystories.tiny_stories_trainer"
CONFIG="fuji-testtrn"
BACKEND="neuron"

COORDINATOR_ADDR="127.0.0.1:5551"
NUM_PROC="1"
PROC_ID="0"

TRAINER_DIR="/tmp/trainer_dir"
DATA_DIR="embedded_in_code"

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

