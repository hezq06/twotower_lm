# Training script for two-tower Bert model
# conda activate ttlm
# A server with 3 Nvidia V100 GPUs is assumed, distributed GPU training is also supported with Deepspeed

deepspeed TwoTowerLM.py --seed 48745 --ds_config ./ds_config/deepspeed_config_24hbert.json --model_type Bert --train_phase phase1_mlm \
--batch_size 128 --window_size 128 --epoch 25 --warm_up_steps 10000 --check_point_path ttbert_base_128_ > log_2towerbert_128.txt

# Add the argument below if you have distributed GPU system.
# --hostfile ./ds_config/myhostfile

pretrain_out_model_128=$(ls ./ttbert_base_128_0_epoch24perp*.cuda:0_final/ttbert_base_128_0_epoch24perp*.cuda:0_final.model)

deepspeed TwoTowerLM.py --seed 38761 --ds_config ./ds_config/deepspeed_config_24hbert.json --model_type Bert --train_phase phase2_t1 \
--pretrained_model $pretrain_out_model_128 --batch_size 256 --window_size 128 --epoch 30 --warm_up_steps 1000 --check_point_path ttbert_base_128_p2_ > log_2towerbert_p2_128.txt

deepspeed TwoTowerLM.py --seed 94761 --ds_config ./ds_config/deepspeed_config_24hbert.json --model_type Bert --train_phase phase3_t2 \
--pretrained_model $pretrain_out_model_128 --batch_size 256 --window_size 128 --epoch 30 --warm_up_steps 1000 --check_point_path ttbert_base_128_p3_ > log_2towerbert_p3_128.txt

