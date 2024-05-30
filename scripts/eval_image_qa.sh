HF_ENDPOINT=https://hf-mirror.com

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ChatUniVi/eval/model_mvchat_bk_vqa.py \
        --model-path=./models/Chat-UniVi \
        --image-folder=../MovieChat/src/output_frame \
        --question-file=../MovieChat/MovieChat-1K-test/annotations \
        --answers-file=./results/breakpoint_${IDX}_of_${CHUNKS}.json \
        --conv-mode=simple \
        --num-chunks=$CHUNKS \
        --chunk-idx=$IDX \
        --temperature=0.2 \
        --num_beams=1 \
        --model_use=BASE \
        --max_new_tokens=1024&
done

wait

echo "All tasks are completed."
