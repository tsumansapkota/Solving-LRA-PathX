#xport CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES="0"

python3 run_tasks.py --model butterfly-64 --task listops --identifier listops_butterfly128 --seed_model 2023 --seed_data 123
python3 run_tasks.py --model butterfly-256 --task text --identifier text_butterfly256 --seed_model 2023 --seed_data 123
python3 run_tasks.py --model butterfly-128 --task retrieval --identifier retrieval_butterfly128 --seed_model 2023 --seed_data 123
python3 run_tasks.py --model butterfly-32 --task image --identifier image_butterfly32 --seed_model 2023 --seed_data 123
python3 run_tasks.py --model butterfly-128 --task pathfinder32-curv_contour_length_14  --identifier pathfinder_butterfly128 --seed_model 2023 --seed_data 123

### Square-root of sequence length (possible lower accuracy)
#python3 run_tasks.py --model butterfly-64 --task listops --identifier listops_butterfly --seed_model 2023 --seed_data 123
#python3 run_tasks.py --model butterfly-64 --task text --identifier text_butterfly --seed_model 2023 --seed_data 123
#python3 run_tasks.py --model butterfly-64 --task retrieval --identifier retrieval_butterfly --seed_model 2023 --seed_data 123
#python3 run_tasks.py --model butterfly-32 --task image --identifier image_butterfly --seed_model 2023 --seed_data 123
#python3 run_tasks.py --model butterfly-32 --task pathfinder32-curv_contour_length_14  --identifier pathfinder_butterfly --seed_model 2023 --seed_data 123