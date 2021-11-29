args=( "$@" )
rank=${args[0]}
cuda_dev_start=${args[1]}
cuda_dev_end=${args[2]}
world_size=${args[3]}
log_dir=${args[4]}
model=${args[5]}
batch_size=${args[@]:6}

for i in $(seq $cuda_dev_start $cuda_dev_end)
do
	numactl --cpunodebind=$(($i < 4 ? 0 : 1)) --membind=$(($i < 4 ? 0 : 1)) \
		python pollux_models_dist_prof.py --master_addr phodgx2 --master_port 5678 --world_size $world_size --rank $rank --device $i --log_dir $log_dir \
		--model $model --batch_size $batch_size &
	rank=$((rank+1))
done
