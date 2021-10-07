for i in {0..3}
do
	numactl --cpunodebind=0 --membind=0 python overlap.py --master_addr phortx2 --master_port 5678 --world_size 8 --rank $i --log_dir ./log/numa_assign > /dev/null& 
done
for i in {4..7}
do
	numactl --cpunodebind=1 --membind=1 python overlap.py --master_addr phortx2 --master_port 5678 --world_size 8 --rank $i --log_dir ./log/numa_assign > /dev/null&
done
