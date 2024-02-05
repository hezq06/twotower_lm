# Training script for two-tower Bert model
# conda activate ttlm
# A server with 3 Nvidia V100 GPUs is assumed, distributed GPU training is also supported with Deepspeed

#export CUDA_VISIBLE_DEVICES=1,2,3

deepspeed TwoTowerLM.py --seed 98753 --ds_config ./ds_config/deepspeed_config_24hbert.json --model_type ttElmo_large --train_phase phase1_mlm \
--batch_size 128 --window_size 128 --epoch 25 --warm_up_steps 10000 --check_point_path ttElmo_large_128_ > log_2towerElmo_large_128.txt

# Add the argument below if you have distributed GPU system.
# --hostfile ./ds_config/myhostfile

pretrain_out_model_128=$(ls ./ttElmo_large_128_epoch24perp*.cuda:0_final/ttElmo_large_128_epoch24perp*.cuda:0_final.model)

deepspeed TwoTowerLM.py --seed 38761 --ds_config ./ds_config/deepspeed_config_24hbert.json --model_type ttElmo_large --train_phase phase2_t1 \
--pretrained_model $pretrain_out_model_128 --batch_size 128 --window_size 128 --epoch 30 --warm_up_steps 1000 --check_point_path ttElmo_large_128_p2_ > log_2towerElmo_p2_128.txt

deepspeed TwoTowerLM.py --seed 94761 --ds_config ./ds_config/deepspeed_config_24hbert.json --model_type ttElmo_large --train_phase phase3_t2 \
--pretrained_model $pretrain_out_model_128 --batch_size 128 --window_size 128 --epoch 30 --warm_up_steps 1000 --check_point_path ttElmo_large_128_p3_ > log_2towerElmo_p3_128.txt

