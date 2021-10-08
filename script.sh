args=( "$@" )
rank=${args[0]}
cuda_dev_start=${args[1]}
cuda_dev_end=${args[2]}
world_size=${args[3]}
log_dir=${args[4]}
batch_size=${args[@]:5}

for i in $(seq $cuda_dev_start $cuda_dev_end)
do
	CUDA_VISIBLE_DEVICES=$i \
		numactl --cpunodebind=$(($i < 4 ? 0 : 1)) --membind=$(($i < 4 ? 0 : 1)) \
		python overlap.py --master_addr phortx2 --master_port 5678 --world_size $world_size --rank $rank --log_dir $log_dir --batch_size $batch_size &
	rank=$((rank+1))
done
