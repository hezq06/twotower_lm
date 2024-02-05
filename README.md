# twotower_lm
Source code for the paper "Spontaneous Emerging Preference in Two-tower Language Model".

# Dedencies
pytorch
deepspeed
pytorch-pretrained-bert
huggingface transformer

# Environment setup
sh env_manage_ttlm.sh
conda activate ttlm
sh install.sh

# Run two-tower langauge modeling
cd workspace
sh run_2towerlm.sh

# Data Preparation
We use Wikitext and Bookcorpus as dataset for pretraining.
We assume users have downloaded the dataset from the web.
Both texts are tokenized using wordpiece tokenizer without case.
We divide Wikitext into 132 blocks, 12 of them for validation. We divide Bookcorpus into 46 blocks and 6 of them for validation.


### Other notice

Deepspeed is using JIT compiling which requires proper version of compilers

For GPT case, you need to hack into
/path/to/home/anaconda3/envs/ttlm/lib/python3.7/site-packages/transformers/models/gpt2/modeling_gpt2.py
line 1091 (GPT2LMHeadModel), add self.hidden_states = hidden_states

The developer name in ther software is a nickname placeholder due to ARR anonymous policy.
 
