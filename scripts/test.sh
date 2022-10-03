# cls + seg + col
CUDA_DEVICE=1
python tools/test.py --trained_dir 'results/REPORT_human/test_score' --focus_head 'refine-f1' --gpu_id $CUDA_DEVICE