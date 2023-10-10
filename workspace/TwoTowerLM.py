"""
TwoTowerBert
Developer: Harry He
"""
from ttlm.dslearn import nca_prepare_training, resume_training
from ttlm.model import TwoTowerDsBert, TwoTowerELMo, TwoTowerGPT
from ttlm.datautil import *

rank=int(os.getenv('LOCAL_RANK', '0'))
allrank = int(os.getenv('RANK', '0'))

args = get_parse_args()
if args.train_phase == "phase1_mlm":
    partition_wiki_train = list(range(allrank, 121, 3))
    partition_book_train = list(range(allrank, 41, 3))
    l_per_epoch_train = 5000 # 5000 default
    l_per_epoch_val = 500 # 500 default
elif args.train_phase in ["phase2_t1", "phase3_t2"]:
    partition_wiki_train = list(range(allrank, 121, 3))
    partition_book_train = list(range(allrank, 41, 3))
    l_per_epoch_train = 1000 # 1000 for step2
    l_per_epoch_val = 100 # 100 for step2
else:
    raise Exception("Unknown train_phase")

if args.model_type == "Bert":
    model_type = TwoTowerDsBert
    bert_config = TwoTowerDsBert.get_model_config(name="large_2tower")
elif args.model_type == "Elmo":
    model_type = TwoTowerELMo
elif args.model_type == "Gpt":
    model_type = TwoTowerGPT
else:
    raise Exception("Unknown model_type")

config={
    "description":"Two tower bert for syntax semantics self emergence",
    "ds_config": args.ds_config,
    "dataset_type":WikiBookDatasetBert,
    "model_type":model_type,
    "pretrained_model":args.pretrained_model,
    "window_size": args.window_size,
    "finetune": False,
    "ds_training": True,
    "batch_size": args.batch_size,
    "epoch": args.epoch,
    "lr_schedule_mode": "linear_decay",
    "lr_linear_decay": 1.0,
    "warm_up_steps": args.warm_up_steps,
    "local_rank": rank,
    "seed": args.seed,
    "train_phase": args.train_phase,
    "data_train":{"mode":"train","max_data_len":l_per_epoch_train*args.batch_size,"partition_wiki":partition_wiki_train,
                  "partition_book":partition_book_train,"window":args.window_size},
    "data_val":{"mode":"val","max_data_len":l_per_epoch_val*args.batch_size,"partition_wiki":list(range(121,133)),"partition_book":list(range(41,47)),
                  "window":args.window_size},
    "check_point_path":args.check_point_path
}

if args.model_type == "Bert":
    config.update(bert_config)

ptM = nca_prepare_training(config)

ptM.run_training()



