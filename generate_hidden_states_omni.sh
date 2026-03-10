source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

TARGET_MODEL_PATH=""
DATA_PATH=""  # path of jsonl file
OUTPUT_PATH=""
NUM_SAMPLES=10000  # change to the exact number of data in DATA_PATH file
AUDIO_ROOT=""

torchrun --nproc_per_node=4 \
    --master_port=29505 \
    scripts/prepare_hidden_states.py \
    --target-model-path $TARGET_MODEL_PATH \
    --enable-aux-hidden-states \
    --data-path $DATA_PATH \
    --chat-template qwen \
    --max-length 2048 \
    --tp-size 4 \
    --batch-size 32 \
    --num-samples $NUM_SAMPLES \
    --output-path $OUTPUT_PATH \
    --is-audio \
    --audio-root $AUDIO_ROOT
