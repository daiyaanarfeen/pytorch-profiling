rank=$1
for i in $(seq $2 $3)
do
	numactl --cpunodebind=$(($i < 4 ? 0 : 1)) --membind=$(($i < 4 ? 0 : 1)) python overlap.py --master_addr phortx3 --master_port 5678 --world_size $4 --rank $rank --log_dir $5 & 
	rank=$((rank+1))
done
