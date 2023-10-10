"""
Python package for bert-style models
for project dynamic masking bert
Developer: Harry He
2022-3-24
"""

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from ttlm.datautil import purge, save_data, load_data, save_model, load_model
from pytorch_pretrained_bert import BertForMaskedLM, BertConfig
from transformers import GPT2LMHeadModel, GPT2Model
import time, copy
from deepspeed import DeepSpeedTransformerConfig, DeepSpeedTransformerLayer

class BertLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        print(
            "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex."
        )

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias

class TwoTowerDsBert(torch.nn.Module):
    """
    A class for wrapping Deepspeed transformer Kernel
    Bert version
    """

    def __init__(self, config):
        super(TwoTowerDsBert, self).__init__()

        self.special_tokens = {"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102, "[MASK]": 103}
        self.config(config)

        self.trf_layers1 = torch.nn.ModuleList([
            copy.deepcopy(DeepSpeedTransformerLayer(self.dsconfig))
            for _ in range(self.num_hidden_layers)
        ])

        self.trf_layers2 = torch.nn.ModuleList([
            copy.deepcopy(DeepSpeedTransformerLayer(self.dsconfig))
            for _ in range(self.num_hidden_layers)
        ])

        self.pt_emb = torch.nn.Embedding(self.output_size, self.tower_num*self.hidden_size)
        self.pt_emb.weight.data.uniform_(-0.02, 0.02)
        self.emb1 = torch.nn.Linear(self.tower_num * self.hidden_size, self.hidden_size)
        self.emb2 = torch.nn.Linear(self.tower_num * self.hidden_size, self.hidden_size)

        self.out_bias = torch.nn.Parameter(torch.zeros(self.output_size))  # For some unknown reason, this size will interfere with transformer kernel, the size should be dividable by 8
        self.input_mask = None
        self.preset_att_mask = None

        self.position_embeddings = torch.nn.Embedding(self.max_seq_length, self.tower_num * self.hidden_size)
        self.position_embeddings.weight.data.uniform_(-0.02, 0.02)
        self.layer_norm_emb = BertLayerNorm(self.tower_num * self.hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.cls_dropout = torch.nn.Dropout(self.dropout_rate)
        self.layer_norm_final = BertLayerNorm(self.tower_num*self.hidden_size)
        self.dense_act = torch.nn.Linear(self.tower_num * self.hidden_size, self.tower_num * self.hidden_size)
        self.gelu = torch.nn.GELU()
        self.layer_norm_trans = BertLayerNorm(self.tower_num * self.hidden_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        if self.train_phase in ["phase2_t1","phase3_t2","phase2_t1t2","phase3_revgrad"]: # phase1_mlm, phase2_t1, phase3_t2, probe_distmat_t1, probe_distmat_t2
            self.dense_act1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.dense_act2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.layer_norm_trans1 = BertLayerNorm(self.hidden_size)
            self.layer_norm_trans2 = BertLayerNorm(self.hidden_size)
            self.outnet1 = torch.nn.Linear(self.hidden_size, self.output_size)
            self.outnet2 = torch.nn.Linear(self.hidden_size, self.output_size)
        elif self.train_phase in ["probe_distmat_t1","probe_distmat_t2"]:
            self.fcp = torch.nn.Linear(self.hidden_size, self.parse_dim, bias=True)
            self.fcp.weight.data.uniform_(-0.02, 0.02)

        self.laboutl = []
        self.lloutl = []
        self.perpvec = []

        self.input_embl = []
        self.encode1l = []
        self.encode2l = []

    def config(self, config):
        self.config_para = config
        self.mask_rate = config.get("mask_rate", 0.15)
        self.replace_by_mask_rate = config.get("replace_by_mask_rate", 0.8)
        self.replace_by_rand_rate = config.get("replace_by_mask_rate", 0.1)
        self.unchange_rate = config.get("replace_by_mask_rate", 0.1)
        self.finetune = config.get("finetune", False)
        self.finetune_output = config.get("finetune_output", 3)
        self.output_size = config.get("output_size", 30528)
        self.mask_id = self.special_tokens["[MASK]"]
        self.pad_id = self.special_tokens["[PAD]"]
        self.cls_id = self.special_tokens["[CLS]"]
        self.sep_id = self.special_tokens["[SEP]"]
        # version control 1 is original with sin position, 2 is after benchmarking with bing_bert
        self.train_phase = config.get("train_phase", "phase1_mlm") # phase2_t1, phase3_t2, probe_distmat_t1, probe_distmat_t2, baseline
        if self.train_phase=="baseline":
            self.tower_num = 1
        else:
            self.tower_num = 2

        self.batch_size = config.get("batch_size", 64)
        self.max_seq_length = config.get("max_seq_length", 512)
        self.window_size = config.get("window_size", 128)
        self.hidden_size = config.get("hidden_size", 1024)
        self.intermediate_size = config.get("intermediate_size", 4096)
        self.model_size = self.hidden_size
        self.heads = config.get("heads", 16)
        self.parse_dim = config.get("parse_dim", 64)
        self.dropout_rate = config.get("dropout_rate", 0.1)
        self.num_hidden_layers = config.get("num_hidden_layers", 24)
        self.local_rank = config.get("local_rank", -1)
        self.seed = config.get("seed", 12345)
        self.fp16 = config.get("fp16", True)
        self.pre_layer_norm = config.get("pre_layer_norm", True)
        self.attn_dropout_checkpoint = config.get("attn_dropout_checkpoint", False)
        self.normalize_invertible = config.get("normalize_invertible", False)
        self.gelu_checkpoint = config.get("gelu_checkpoint", False)
        self.stochastic_mode = config.get("stochastic_mode", False)
        self.ds_training = config.get("ds_training", True) ## DeepspeedTransformer Kernel doesn't work with eval()

        self.dsconfig = DeepSpeedTransformerConfig(
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            heads=self.heads,
            attn_dropout_ratio=self.dropout_rate,
            hidden_dropout_ratio=self.dropout_rate,
            num_hidden_layers=self.num_hidden_layers,
            initializer_range=0.02,
            local_rank=self.local_rank,
            seed=self.seed,
            fp16=self.fp16,
            pre_layer_norm=self.pre_layer_norm,
            attn_dropout_checkpoint=self.attn_dropout_checkpoint,
            normalize_invertible=self.normalize_invertible,
            gelu_checkpoint=self.gelu_checkpoint,
            stochastic_mode=self.stochastic_mode,
            training=self.ds_training)

    @staticmethod
    def get_model_config(name="large"):

        config = {}

        if name == "small":
            config["num_hidden_layers"] = 4
            config["heads"] = 8
            config["hidden_size"] = 512
            config["intermediate_size"] = 2048
        elif name == "medium":
            config["num_hidden_layers"] = 8
            config["heads"] = 8
            config["hidden_size"] = 512
            config["intermediate_size"] = 2048
        elif name == "base":
            config["num_hidden_layers"] = 12
            config["heads"] = 12
            config["hidden_size"] = 768
            config["intermediate_size"] = 3072
        elif name == "large":
            config["num_hidden_layers"] = 24
            config["heads"] = 16
            config["hidden_size"] = 1024
            config["intermediate_size"] = 4096
        elif name == "large_2tower":
            config["num_hidden_layers"] = 24
            config["heads"] = 16
            config["hidden_size"] = 512
            config["intermediate_size"] = 2048
        elif name == "base_2tower":
            config["num_hidden_layers"] = 12
            config["heads"] = 12
            config["hidden_size"] = 384
            config["intermediate_size"] = 1024
        else:
            raise Exception("Unknown model config.")

        return config

    def input_masking(self, inputx):
        """
        never mask <s> and </s>
        replace_by_mask_rate -> <mask> 30003
        replace_by_rand_rate -> <rand> != <s>, </s>, </pad>, </mask>
        unchange_rate -> <rand>
        :return:
        """
        ## rnd for mask
        rnd = torch.rand((inputx.shape[0], inputx.shape[1]), device=self.cuda_device)
        self.input_mask = torch.zeros((inputx.shape[0], inputx.shape[1]), device=self.cuda_device)
        unmaskbool = torch.logical_or(rnd > self.mask_rate, inputx == self.cls_id)  # not mask for [CLS], [SEP], [PAD]
        unmaskbool = torch.logical_or(unmaskbool, inputx == self.sep_id)
        unmaskbool = torch.logical_or(unmaskbool, inputx == self.pad_id)
        self.input_mask[unmaskbool] = 1  # 1 means not mask

        ## rnd for branch <mask>,<rand>,<unchanged>
        rnd = torch.rand((inputx.shape[0], inputx.shape[1]), device=self.cuda_device)
        maskid = rnd > (1 - self.replace_by_mask_rate)
        inputx[torch.logical_and(maskid, torch.logical_not(unmaskbool))] = self.mask_id
        rndwrd = torch.rand((inputx.shape[0], inputx.shape[1]), device=self.cuda_device) * self.output_size
        rndwrd = torch.floor(rndwrd).type(torch.cuda.LongTensor)
        randid = torch.logical_and(rnd < (1 - self.replace_by_mask_rate),
                                   rnd > (1 - self.replace_by_mask_rate - self.replace_by_rand_rate))
        inputx[torch.logical_and(randid, torch.logical_not(unmaskbool))] = rndwrd[
            torch.logical_and(randid, torch.logical_not(unmaskbool))]

        return inputx

    def cal_att_masking(self, inputx, finetune_flag):
        att_mask = (inputx != self.pad_id).type(torch.LongTensor).to(self.cuda_device).unsqueeze(1).unsqueeze(2)
        att_mask = att_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        att_mask = (1.0 - att_mask) * -10000.0
        return att_mask

    # @torch.cuda.amp.autocast()
    def forward(self, inputx, schedule=None):
        """
        Forward
        # Transformer, use [batch, seq_len, hd_size] convention
        :param inputx: [batch, seq_len] of IDs
        :param hidden:
        :return:
        """
        try:
            self.cuda_device = inputx.device
        except:
            self.cuda_device = inputx[0].device

        ## DS convention, 0 is masked out for padding, shape
        if not self.finetune:
            inputx = self.input_masking(inputx)

        if self.train_phase == "baseline":
            # Single tower baseline
            att_mask = self.cal_att_masking(inputx, self.finetune)
            seq_length = inputx.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputx.device)
            position_ids = position_ids.unsqueeze(0).expand_as(inputx)
            self.input_emb = self.pt_emb(inputx)
            enc_output = self.input_emb + self.position_embeddings(position_ids)
            enc_output = self.layer_norm_emb(enc_output)
            enc_output = self.dropout(enc_output)

            enc_output = self.emb1(enc_output)

            for trf_l in self.trf_layers1:
                enc_output = trf_l(enc_output, attention_mask=att_mask)

            self.enc_output = enc_output

            enc_output = self.layer_norm_final(enc_output)
            enc_output = self.gelu(self.dense_act(enc_output))
            enc_output = self.layer_norm_trans(enc_output)
            output = F.linear(enc_output, self.pt_emb.weight)
            output = output + self.out_bias[:self.output_size]
            return output

        if self.train_phase == "phase1_mlm":

            att_mask = self.cal_att_masking(inputx, self.finetune)
            seq_length = inputx.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputx.device)
            position_ids = position_ids.unsqueeze(0).expand_as(inputx)
            self.input_emb = self.pt_emb(inputx)
            enc_output = self.input_emb + self.position_embeddings(position_ids)
            enc_output = self.layer_norm_emb(enc_output)
            enc_output = self.dropout(enc_output)

            enc_output1 = self.emb1(enc_output)
            enc_output2 = self.emb2(enc_output)

            for trf_l in self.trf_layers1:
                enc_output1 = trf_l(enc_output1, attention_mask=att_mask)

            for trf_l in self.trf_layers2:
                enc_output2 = trf_l(enc_output2, attention_mask=att_mask)

            self.enc_output1 = enc_output1
            self.enc_output2 = enc_output2

            enc_output = torch.cat([enc_output1,enc_output2],dim=-1)
            enc_output = self.layer_norm_final(enc_output)

            enc_output = self.gelu(self.dense_act(enc_output))
            enc_output = self.layer_norm_trans(enc_output)
            output = F.linear(enc_output, self.pt_emb.weight)
            output = output + self.out_bias[:self.output_size]
            return output

        elif self.train_phase in ["phase2_t1","phase3_t2"]:
            att_mask = self.cal_att_masking(inputx, self.finetune)

            with torch.no_grad():

                seq_length = inputx.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=inputx.device)
                position_ids = position_ids.unsqueeze(0).expand_as(inputx)
                enc_output = self.pt_emb(inputx) + self.position_embeddings(position_ids)
                enc_output = self.layer_norm_emb(enc_output)
                enc_output = self.dropout(enc_output)

                enc_output1 = self.emb1(enc_output)
                enc_output2 = self.emb2(enc_output)
                self.enc_output_l = [[enc_output1],[enc_output2]]

                for trf_l in self.trf_layers1:
                    enc_output1 = trf_l(enc_output1, attention_mask=att_mask)
                    self.enc_output_l[0].append(enc_output1)

                for trf_l in self.trf_layers2:
                    enc_output2 = trf_l(enc_output2, attention_mask=att_mask)
                    self.enc_output_l[1].append(enc_output2)

                enc_output = torch.cat([enc_output1, enc_output2], dim=-1)
                enc_output = self.layer_norm_final(enc_output)

            if self.train_phase == "phase2_t1":

                enc_output = self.gelu(self.dense_act1(enc_output[:,:,:self.hidden_size]))
                enc_output = self.layer_norm_trans1(enc_output)
                output = self.outnet1(enc_output)
                self.output = output
                return output

            elif self.train_phase == "phase3_t2":

                enc_output = self.gelu(self.dense_act2(enc_output[:,:,self.hidden_size:]))
                enc_output = self.layer_norm_trans2(enc_output)
                output = self.outnet2(enc_output)
                self.output = output
                return output

        elif self.train_phase in ["probe_distmat_t1", "probe_distmat_t2"]:
            assert self.finetune == True
            output = self.forward_distmat(inputx)
            return output

        else:
            raise Exception("Unknown train_phase")

    def forward_distmat(self, inputx):

        inputd = inputx[0]
        att_mask = inputx[1]
        att_mask = att_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        att_mask = (1.0 - att_mask) * -10000.0

        with torch.no_grad():

            seq_length = inputd.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputd.device)
            position_ids = position_ids.unsqueeze(0).expand_as(inputd)
            enc_output = self.pt_emb(inputd) + self.position_embeddings(position_ids)
            enc_output = self.layer_norm_emb(enc_output)
            enc_output = self.dropout(enc_output)

            enc_output1 = self.emb1(enc_output)
            enc_output2 = self.emb2(enc_output)
            self.enc_output_l = [[enc_output1], [enc_output2]]

            for trf_l in self.trf_layers1:
                enc_output1 = trf_l(enc_output1, attention_mask=att_mask)
                self.enc_output_l[0].append(enc_output1)

            for trf_l in self.trf_layers2:
                enc_output2 = trf_l(enc_output2, attention_mask=att_mask)
                self.enc_output_l[1].append(enc_output2)

            enc_output = torch.cat([enc_output1, enc_output2], dim=-1)
            enc_output = self.layer_norm_final(enc_output)

        if self.train_phase=="probe_distmat_t1":
            # enc_output = self.fcp(enc_output[:, :, :self.hidden_size])
            enc_output = self.fcp(self.enc_output_l[0][8])

        elif self.train_phase=="probe_distmat_t2":
            # enc_output = self.fcp(enc_output[:, :, :self.hidden_size])
            enc_output = self.fcp(self.enc_output_l[1][8])

        else:
            raise Exception("Unknown train_phase")

        x = enc_output.unsqueeze(2)
        x = x.expand(-1, -1, self.window_size, -1)
        xt = x.transpose(1, 2)
        dists = x - xt
        sq_dists = dists.pow(2)
        sq_dists = torch.sqrt(torch.sum(sq_dists, -1) + 1e-4)
        return sq_dists

    def cal_loss(self, output, labels):
        if not self.finetune:
            loss = CAL_LOSS.Masked_CrossEntropyLoss(output, labels, self.input_mask)
            self.output = output
            self.labels = labels
        elif self.train_phase in ["probe_distmat_t1", "probe_distmat_t2"]:
            input_mask = (labels == -1).to(dtype=next(self.parameters()).dtype)   # -1 is dist mask
            loss_mse = torch.nn.MSELoss(reduce=False)
            loss = loss_mse(output, labels.to(dtype=next(self.parameters()).dtype))
            lossup = torch.mean(loss * (1 - input_mask))
            lossdown = torch.mean(1 - input_mask)
            loss = lossup / lossdown
        else:
            loss = CAL_LOSS.CrossEntropyLoss(output, labels)
        return loss

    def prepare_optimizer_parameters(self, weight_decay=0.01):

        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'attn_nw', 'attn_nb', 'norm_w', 'norm_b',
                    'attn_qkvb', 'attn_ob',
                    'inter_b', 'output_b']

        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                weight_decay
        }, {
            'params':
                [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }]

        return optimizer_grouped_parameters

    def eval_mem(self, datax, labels): # append predicted perplexity of each word

        self.input_embl.append(self.input_emb[labels != 0])
        self.laboutl.append(labels[labels != 0])
        # self.laboutl.append(labels[self.input_mask == 0])
        # self.encode1l.append(self.enc_output1[self.input_mask == 0])
        # self.encode2l.append(self.enc_output2[self.input_mask == 0])

    def post_eval_mem(self):
        self.input_embl = torch.cat(self.input_embl)
        self.laboutl = torch.cat(self.laboutl)
        # self.encode1l = torch.cat(self.encode1l)
        # self.encode2l = torch.cat(self.encode2l)

    def eval_mem_1(self, datax, labels): # append predicted perplexity of each word

        lloutput = self.softmax(self.output)
        plabel = labels[self.input_mask == 0]
        pout = lloutput[self.input_mask == 0]
        llout = torch.gather(pout,1,plabel.view(-1,1)).squeeze()
        self.laboutl.append(plabel)
        self.lloutl.append(llout)
        # endt = time.time()
        # print(endt-stt)

    def post_eval_mem_1(self):
        laboutl = torch.cat(self.laboutl)
        lloutl = torch.cat(self.lloutl)

        self.learndict = {}
        laboutl = purge(laboutl)
        lloutl = purge(lloutl)
        for ii in range(len(laboutl)):
            lb = laboutl[ii]
            if lb in self.learndict.keys():
                self.learndict[lb].append(lloutl[ii])
            else:
                self.learndict[lb] = [lloutl[ii]]

        for key, val in self.learndict.items():
            self.learndict[key]=np.mean(val)

        self.perpvec = np.zeros(self.output_size)
        for key, val in self.learndict.items():
            self.perpvec[key] = val

class TwoTowerELMo(torch.nn.Module):
    """
    A class for wrapping Deepspeed transformer Kernel
    Bert version
    """

    def __init__(self, config):
        super(self.__class__, self).__init__()

        self.special_tokens = {"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102, "[MASK]": 103}
        self.config(config)

        self.pt_emb = torch.nn.Embedding(self.output_size, 2*self.model_size)
        self.pt_emb.weight.data.uniform_(-0.02, 0.02)
        self.emb1 = torch.nn.Linear(2 * self.model_size, self.model_size)
        self.emb2 = torch.nn.Linear(2 * self.model_size, self.model_size)
        self.out_bias = torch.nn.Parameter(torch.zeros(self.output_size))
        self.layer_norm_embf = BertLayerNorm(2*self.model_size)
        self.layer_norm_embb = BertLayerNorm(2*self.model_size)

        self.lstm_forward1 = torch.nn.LSTM(self.model_size, self.hidden_size, num_layers=self.num_hidden_layers,
                                          batch_first=True)
        self.lstm_backward1 = torch.nn.LSTM(self.model_size, self.hidden_size, num_layers=self.num_hidden_layers,
                                           batch_first=True)
        self.h2o_f1 = torch.nn.Linear(self.hidden_size, self.model_size)
        self.h2o_b1 = torch.nn.Linear(self.hidden_size, self.model_size)

        self.lstm_forward2 = torch.nn.LSTM(self.model_size, self.hidden_size, num_layers=self.num_hidden_layers,
                                           batch_first=True)
        self.lstm_backward2 = torch.nn.LSTM(self.model_size, self.hidden_size, num_layers=self.num_hidden_layers,
                                            batch_first=True)
        self.h2o_f2 = torch.nn.Linear(self.hidden_size, self.model_size)
        self.h2o_b2 = torch.nn.Linear(self.hidden_size, self.model_size)

        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.layer_norm_final = BertLayerNorm(4 * self.model_size)
        self.dense_act = torch.nn.Linear(4 * self.model_size, 2*self.model_size)
        self.gelu = torch.nn.GELU()
        self.layer_norm_trans = BertLayerNorm(2*self.model_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        if self.train_phase in ["phase2_t1","phase3_t2"]:
            self.dense_act1 = torch.nn.Linear(2*self.model_size, self.model_size)
            self.dense_act2 = torch.nn.Linear(2*self.model_size, self.model_size)
            self.layer_norm_trans1 = BertLayerNorm(self.model_size)
            self.layer_norm_trans2 = BertLayerNorm(self.model_size)
            self.outnet1 = torch.nn.Linear(self.model_size, self.output_size)
            self.outnet2 = torch.nn.Linear(self.model_size, self.output_size)

        self.laboutl = []
        self.lloutl = []
        self.perpvec = []
        self.input_embl = []

    def config(self,config):
        self.config_para = config
        self.finetune = config.get("finetune", False)
        self.output_size = config.get("output_size", 30528)
        self.mask_id = self.special_tokens["[MASK]"]
        self.pad_id = self.special_tokens["[PAD]"]
        self.cls_id = self.special_tokens["[CLS]"]
        self.sep_id = self.special_tokens["[SEP]"]
        self.train_phase = config.get("train_phase", "phase1_mlm")  # phase2_t1, phase3_t2

        self.batch_size = config.get("batch_size", 64)
        self.window_size = config.get("window_size", 128)
        self.hidden_size = config.get("hidden_size", 2048)
        self.model_size = config.get("model_size", 256)
        self.num_hidden_layers = config.get("num_hidden_layers", 2)
        self.dropout_rate = config.get("dropout_rate", 0.1)

    def flip_seq(self, tensor):
        """
        not inplace flip of window dimension
        :param tensor:
        :return:
        """
        flipt = torch.eye(self.window_size).type(tensor.type()).to(self.cuda_device)
        flipt = torch.flip(flipt, [0])
        tensor = torch.einsum("bwl,ww->bwl", tensor, flipt)
        return tensor

    def forward_tt(self,inputx):

        self.cuda_device = inputx.device
        batch, length = inputx.shape
        ### Input data forward, pad on the left
        inputx_f = torch.cat([torch.zeros((batch, 1)).type(torch.LongTensor).to(self.cuda_device), inputx[:, :-1]],
                             dim=1)
        ### Input data backward, flip and pad on the left
        inputx_b = torch.flip(inputx, [1])
        inputx_b = torch.cat([torch.zeros((batch, 1)).type(torch.LongTensor).to(self.cuda_device), inputx_b[:, :-1]],
                             dim=1)

        enc_output_f = self.pt_emb(inputx_f)
        enc_output_f = self.layer_norm_embf(enc_output_f)
        enc_output_f = self.dropout(enc_output_f)

        enc_output_b = self.pt_emb(inputx_b)
        enc_output_b = self.layer_norm_embb(enc_output_b)
        enc_output_b = self.dropout(enc_output_b)

        enc_output_f1 = self.emb1(enc_output_f)
        enc_output_f2 = self.emb2(enc_output_f)
        enc_output_b1 = self.emb1(enc_output_b)
        enc_output_b2 = self.emb2(enc_output_b)

        hout_f1, _ = self.lstm_forward1(enc_output_f1)
        output_f1 = self.h2o_f1(hout_f1)

        hout_b1, _ = self.lstm_backward1(enc_output_b1)
        output_b1 = self.h2o_b1(hout_b1)
        output_b1 = self.flip_seq(output_b1)

        hout_f2, _ = self.lstm_forward2(enc_output_f2)
        output_f2 = self.h2o_f2(hout_f2)

        hout_b2, _ = self.lstm_backward2(enc_output_b2)
        output_b2 = self.h2o_b2(hout_b2)
        output_b2 = self.flip_seq(output_b2)

        enc_output = torch.cat([output_f1, output_b1, output_f2, output_b2], dim=-1)
        enc_output = self.layer_norm_final(enc_output)
        enc_output = self.dropout(enc_output)
        return enc_output


    def forward(self, inputx, schedule=None):
        """
        Forward
        # Transformer, use [batch, seq_len, hd_size] convention
        :param inputx: [batch, seq_len] of IDs
        :param hidden:
        :return:
        """
        self.input_emb = self.pt_emb(inputx)

        if self.train_phase == "phase1_mlm":

            enc_output = self.forward_tt(inputx)

            enc_output = self.dense_act(enc_output)
            enc_output = self.gelu(enc_output)
            enc_output = self.layer_norm_trans(enc_output)
            output = F.linear(enc_output, self.pt_emb.weight)
            output = output + self.out_bias[:self.output_size]

            return output

        elif self.train_phase in ["phase2_t1","phase3_t2"]:

            with torch.no_grad():
                enc_output = self.forward_tt(inputx)

            if self.train_phase == "phase2_t1":

                enc_output = self.dense_act1(enc_output[:, :, :2*self.model_size])
                enc_output = self.gelu(enc_output)
                enc_output = self.layer_norm_trans1(enc_output)
                output = self.outnet1(enc_output)
                self.output = output
                return output

            elif self.train_phase == "phase3_t2":

                enc_output = self.dense_act2(enc_output[:, :, 2*self.model_size:])
                enc_output = self.gelu(enc_output)
                enc_output = self.layer_norm_trans2(enc_output)
                output = self.outnet2(enc_output)
                self.output = output
                return output

        else:
            raise Exception("Unknown train_phase")

    def eval_mem(self, datax, labels): # append predicted perplexity of each word

        self.input_embl.append(self.input_emb[labels != 0])
        self.laboutl.append(labels[labels != 0])
        # self.laboutl.append(labels[self.input_mask == 0])
        # self.encode1l.append(self.enc_output1[self.input_mask == 0])
        # self.encode2l.append(self.enc_output2[self.input_mask == 0])

    def post_eval_mem(self):
        self.input_embl = torch.cat(self.input_embl)
        self.laboutl = torch.cat(self.laboutl)
        # self.encode1l = torch.cat(self.encode1l)
        # self.encode2l = torch.cat(self.encode2l)

    # def eval_mem(self, datax, labels): # append predicted perplexity of each word
    #
    #     lloutput = self.softmax(self.output)
    #     llout = torch.gather(lloutput.view(-1,self.output_size),1,labels.view(-1,1)).squeeze()
    #     self.laboutl.append(labels)
    #     self.lloutl.append(llout)
    #
    # def post_eval_mem(self):
    #     laboutl = torch.cat(self.laboutl).view(-1)
    #     lloutl = torch.cat(self.lloutl)
    #
    #     self.learndict = {}
    #     laboutl = purge(laboutl)
    #     lloutl = purge(lloutl)
    #     for ii in range(len(laboutl)):
    #         lb = laboutl[ii]
    #         if lb in self.learndict.keys():
    #             self.learndict[lb].append(lloutl[ii])
    #         else:
    #             self.learndict[lb] = [lloutl[ii]]
    #
    #     for key, val in self.learndict.items():
    #         self.learndict[key] = np.mean(val)
    #
    #     self.perpvec = np.zeros(self.output_size)
    #     for key, val in self.learndict.items():
    #         self.perpvec[key] = val

    def cal_loss(self, output, labels):
        loss = CAL_LOSS.CrossEntropyLoss(output, labels)
        return loss

class TwoTowerGPT(torch.nn.Module):
    """
    A class for wrapping Deepspeed transformer Kernel
    Bert version
    """

    def __init__(self, config):
        super(self.__class__, self).__init__()

        self.special_tokens = {"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102, "[MASK]": 103}
        self.config(config)

        gpt_config = self.gpt_config()
        self.model1 = GPT2LMHeadModel(gpt_config)
        self.model2 = GPT2LMHeadModel(gpt_config)

        self.model2.transformer.wte = self.model1.transformer.wte
        self.model2.transformer.wpe = self.model1.transformer.wpe
        self.out_bias = torch.nn.Parameter(torch.zeros(self.output_size))  # For some unknown reason, this size will interfere with transformer kernel, the size should be dividable by 8

        self.layer_norm_final = BertLayerNorm(2 * self.hidden_size)
        self.dense_act = torch.nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.gelu = torch.nn.GELU()
        self.layer_norm_trans = BertLayerNorm(self.hidden_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        if self.train_phase in ["phase2_t1","phase3_t2"]:
            self.dense_act1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.dense_act2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.layer_norm_trans1 = BertLayerNorm(self.hidden_size)
            self.layer_norm_trans2 = BertLayerNorm(self.hidden_size)
            self.outnet1 = torch.nn.Linear(self.hidden_size, self.output_size)
            self.outnet2 = torch.nn.Linear(self.hidden_size, self.output_size)

        self.laboutl = []
        self.lloutl = []
        self.perpvec = []
        self.input_embl=[]

    def config(self, config):
        self.config_para = config
        self.train_phase = config.get("train_phase", "phase1_mlm") # phase2_t1, phase3_t2
        self.window_size = config.get("window_size", 128)
        self.hidden_size = config.get("hidden_size", 384)
        self.pretrained_gpt = config.get("pretrained_gpt", False)
        self.output_size = config.get("output_size", 30528)  # fuse, seperate
        self.mask_id = self.special_tokens["[MASK]"]
        self.pad_id = self.special_tokens["[PAD]"]
        self.cls_id = self.special_tokens["[CLS]"]
        self.sep_id = self.special_tokens["[SEP]"]

    def gpt_config(self):
        config = GPT2LMHeadModel.from_pretrained('gpt2').config
        config.bos_token_id = self.special_tokens["[CLS]"]
        config.eos_token_id = self.special_tokens["[SEP]"]
        config.vocab_size = self.output_size
        config.n_positions = self.window_size
        config.n_embd = self.hidden_size
        config.return_dict = False
        return config

    def attention_mask(self, inputx):
        ## Attention mask for padding
        att_mask = (inputx != self.special_tokens["[PAD]"]) # [b, w]
        att_mask = att_mask.type(torch.LongTensor).to(self.cuda_device)
        return att_mask

    # @torch.cuda.amp.autocast()
    def forward(self, inputx, schedule=None):
        """
        Forward
        # Transformer, use [batch, seq_len, hd_size] convention
        :param inputx: [batch, seq_len] of IDs
        :param hidden:
        :return:
        """
        self.cuda_device = inputx.device
        att_mask = self.attention_mask(inputx)
        label_in = copy.deepcopy(inputx)
        label_in[inputx == 0] = -100  # -100 is ignored in GPT

        if self.train_phase == "phase1_mlm":

            loss_all1 = self.model1(input_ids=inputx, attention_mask=att_mask, labels=label_in)
            loss_all2 = self.model2(input_ids=inputx, attention_mask=att_mask, labels=label_in)

            self.input_emb = self.model1.transformer.wte(inputx)

            enc_output1 = self.model1.hidden_states
            enc_output2 = self.model2.hidden_states
            # enc_output1 = loss_all1.hidden_states
            # enc_output2 = loss_all2.hidden_states

            enc_output = torch.cat([enc_output1,enc_output2],dim=-1)
            enc_output = self.layer_norm_final(enc_output)

            enc_output = self.gelu(self.dense_act(enc_output))
            enc_output = self.layer_norm_trans(enc_output)
            output = F.linear(enc_output, self.model1.transformer.wte.weight)
            output = output + self.out_bias[:self.output_size]
            return output

        elif self.train_phase in ["phase2_t1","phase3_t2"]:

            with torch.no_grad():

                loss_all1 = self.model1(input_ids=inputx, attention_mask=att_mask, labels=label_in)
                loss_all2 = self.model2(input_ids=inputx, attention_mask=att_mask, labels=label_in)

                # hack into /home/hezq17/anaconda3/envs/dynmlm/lib/python3.7/site-packages/transformers/models/gpt2/modeling_gpt2.py
                # line 920 (GPT2LMHeadModel), add self.hidden_states = hidden_states
                enc_output1 = self.model1.hidden_states
                enc_output2 = self.model2.hidden_states
                # enc_output1 = loss_all1.hidden_states
                # enc_output2 = loss_all2.hidden_states

                enc_output = torch.cat([enc_output1, enc_output2], dim=-1)
                enc_output = self.layer_norm_final(enc_output)

            if self.train_phase == "phase2_t1":

                enc_output = self.gelu(self.dense_act1(enc_output[:,:,:self.hidden_size]))
                enc_output = self.layer_norm_trans1(enc_output)
                output = self.outnet1(enc_output)
                self.output = output
                return output

            elif self.train_phase == "phase3_t2":

                enc_output = self.gelu(self.dense_act2(enc_output[:,:,self.hidden_size:]))
                enc_output = self.layer_norm_trans2(enc_output)
                output = self.outnet2(enc_output)
                self.output = output
                return output

        else:
            raise Exception("Unknown train_phase")


    def cal_loss(self, output, labels):
        # Shift so that tokens < n predict n
        shift_output = output[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = CAL_LOSS.CrossEntropyLoss(shift_output, shift_labels)
        return loss

    def eval_mem(self, datax, labels): # append predicted perplexity of each word

        self.input_embl.append(self.input_emb[labels != 0])
        self.laboutl.append(labels[labels != 0])
        # self.laboutl.append(labels[self.input_mask == 0])
        # self.encode1l.append(self.enc_output1[self.input_mask == 0])
        # self.encode2l.append(self.enc_output2[self.input_mask == 0])

    def post_eval_mem(self):
        self.input_embl = torch.cat(self.input_embl)
        self.laboutl = torch.cat(self.laboutl)
        # self.encode1l = torch.cat(self.encode1l)
        # self.encode2l = torch.cat(self.encode2l)


class CAL_LOSS(torch.nn.Module):
    """
    An abstract loss wrapper
    model must have a cal_loss method
    """

    def __init__(self, model):
        super(self.__class__, self).__init__()
        self.model = model

    # @torch.cuda.amp.autocast()
    def forward(self, datax, labels, schedule=1.0):

        output = self.model(datax, schedule=schedule)
        self.output = output
        loss = self.model.cal_loss(output, labels)

        # print("Original Loss,",loss)
        if self.loss_mode=="train":
            loss = self.cal_loss_reg_recursize(self.model, loss)
        # print("After reg Loss,", loss)

        return loss

    def cal_loss_reg_recursize(self, model, loss):
        if hasattr(model, "loss_reg"):
            # print(type(model))
            loss = loss + model.loss_reg
        if hasattr(model, "submodels"):
            for submodel in model.submodels:
                loss = self.cal_loss_reg_recursize(submodel, loss)
        return loss

    @staticmethod
    def CrossEntropyLoss(output, labels):
        device = labels.device
        lossc = torch.nn.CrossEntropyLoss(reduction='none')
        if len(output.shape)==2:
            loss = lossc(output, labels.type(torch.LongTensor).squeeze(dim=-1).to(device))
        elif len(output.shape)==3:
            loss = lossc(output.permute(0,2,1), labels.type(torch.LongTensor).to(device))
        return torch.mean(loss)

    @staticmethod
    def Masked_CrossEntropyLoss(output, labels, input_mask):
        """
        transformer style masked loss
        :param output: [batch, w, l_size]
        :param labels:
        :param input_mask:
        :return:
        """
        lossc=torch.nn.CrossEntropyLoss(reduction='none',ignore_index=-1)
        assert len(output.shape)==3
        outputpm = output.permute(0, 2, 1)
        if outputpm.shape[-1] != labels.shape[-1] and labels.shape[-1]==1:
            labels = labels.expand([-1,outputpm.shape[-1]])
        loss = lossc(outputpm, labels.type(torch.cuda.LongTensor))
        lossm = torch.sum(loss*(1-input_mask))/(torch.sum(1-input_mask)) # 1 means not predict
        return lossm

