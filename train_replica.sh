exp_name='baseline'

voxel_size=0.001
update_init_factor=16
appearance_dim=0
ratio=1

ulimit -n 4096

# ./train.sh -d replica/office_0/Sequence_1 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/office_0/Sequence_2 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/office_1/Sequence_1 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/office_1/Sequence_2 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/office_2/Sequence_1 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/office_2/Sequence_2 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/office_3/Sequence_1 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/office_3/Sequence_2 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/office_4/Sequence_1 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/office_4/Sequence_2 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
./train.sh -d replica/room_0/Sequence_1 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/room_0/Sequence_2 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/room_1/Sequence_1 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/room_1/Sequence_2 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/room_2/Sequence_1 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s
# 
# ./train.sh -d replica/room_2/Sequence_2 -l ${exp_name} --gpu -1 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} #& 
# sleep 20s

