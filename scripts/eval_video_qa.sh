HF_ENDPOINT=https://hf-mirror.com

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ChatUniVi/eval/model_mvchat_gb_vqa.py \
        --model-path=./models/Chat-UniVi \
        --video-folder=../MovieChat/MovieChat-1K-test/videos \
        --question-file=../MovieChat/MovieChat-1K-test/annotations \
        --answers-file=./results/global_${IDX}_of_${CHUNKS}.json \
        --num-chunks=$CHUNKS \
        --chunk-idx=$IDX \
        --max_frames=13000&
done

wait

echo "All tasks are completed."
