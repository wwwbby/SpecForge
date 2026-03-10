source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

PROXY_FILE_PATH=""
source $PROXY_FILE_PATH
unset http_proxy
unset https_proxy
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

TARGET_MODEL_PATH=""
PORT=8001
INPUT_FILE_PATH=""
OUTPUT_FILE_PATH=""

python scripts/regenerate_train_data.py \
    --model ${TARGET_MODEL_PATH} \
    --concurrency 128 \
    --max-tokens 2048 \
    --server-address 127.0.0.1:${PORT} \
    --temperature 0 \
    --input-file-path $INPUT_FILE_PATH \
    --output-file-path $OUTPUT_FILE_PATH
