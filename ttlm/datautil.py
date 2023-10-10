"""
Python package for data handling
for project dynamic masking bert
Developer: Harry He
2022-3-25
"""
import pickle, json, os
import numpy as np
import collections
import torch
import argparse
from pytorch_pretrained_bert import BertTokenizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
try:
    from wordcloud import WordCloud
except:
    pass

def save_data(data,file,large_data=False, engine="pickle"):
    if engine=="pickle":
        if not large_data:
            pickle.dump(data, open(file, "wb"))
            print("Data saved to ", file)
        else:
            pickle.dump(data, open(file, "wb"), protocol=4)
            print("Large Protocal 4 Data saved to ", file)
    elif engine=="json":
        json.dump(data, open(file, "w"))
        print("Data saved to ", file)
    else:
        raise Exception("Unknown Engine.")

def load_data(file, engine="pickle", print_flag=True):
    if engine == "pickle":
        data = pickle.load(open(file, "rb"))
        if print_flag:
            print("Data load from ", file)
        return data
    elif engine=="json":
        data = json.load(open(file, "r"))
        if print_flag:
            print("Data load from ", file)
        return data
    else:
        raise Exception("Unknown Engine.")

def save_text(string, file_name):

    with open(file_name,"w") as f:
        f.write(string)

def save_model(model,file):
    torch.save(model, file)

def load_model(file,map_location=None):
    return torch.load(file, map_location=map_location)

def copy_model_state(model, model_from, except_list=[]):

    try:
        model.load_state_dict(model_from.state_dict())
        print("Model copied.")
    except Exception as inst:
        print(inst)
        print("Try Flexible model loading ...")
        pretrained_dict = model_from.state_dict()
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (k not in except_list)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model

def purge(tensor):
    try:
        tensor=tensor.detach()
    except:
        pass
    tensor=tensor.cpu().numpy()
    return tensor

def get_parse_args():
    ## Source code partially copy from deepspeed bingbert example

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_config",
        default="deepspeed_config.json",
        type=str,
        help="Pointer to the configuration file of the experiment."
    )
    parser.add_argument(
        "--pretrained_model",
        default=None,
        type=str,
        help="Pretrained model to start with."
    )
    parser.add_argument(
            "--batch_size",
            default=64,
            type=int,
            help=
            "Batch size,make sure to match this with the batch size in ds_config. "
        )
    parser.add_argument(
            "--window_size",
            default=128,
            type=int,
            help=
            "The maximum total input sequence length after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and sequences shorter than this will be padded."
        )
    parser.add_argument(
        '--seed',
        type=int,
        default=12345,
        help="Random seed for initialization."
    )
    parser.add_argument(
        '--warm_up_steps',
        type=int,
        default=10000,
        help="Warm up steps."
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=2,
        help="Number of epochs."
    )
    parser.add_argument(
        '--check_point_path',
        type=str,
        default="debug_",
        help="Check point path."
    )
    parser.add_argument( # deep speed seems to append this arg to the python file
        '--local_rank',
        type=int,
        default=-1,
        help="Local rank"
    )
    parser.add_argument(
        '--train_phase',
        type=str,
        default="phase1_mlm",
        help="Train phase (phase1_mlm, phase2_t1, phase3_t2)"
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default="Bert",
        help="Model type (Bert, Elmo or Gpt)"
    )

    args = parser.parse_args()

    return args

def plot_mat(data,start=0,lim=1000,symmetric=False,title=None,tick_step=None,show=True,xlabel=None,ylabel=None, clim=None):
    if show:
        plt.figure()
    data=np.array(data)
    if len(data.shape) != 2:
        data=data.reshape(1,-1)
    img=data[:,start:start+lim]
    if symmetric:
        plt.imshow(img, cmap='seismic',clim=(-np.amax(np.abs(data)), np.amax(np.abs(data))))
        # plt.imshow(img, cmap='seismic', clim=(-2,2))
    else:
        plt.imshow(img, cmap='seismic',clim=clim)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    if tick_step is not None:
        plt.xticks(np.arange(0, len(img[0]), tick_step))
        plt.yticks(np.arange(0, len(img), tick_step))
    if xlabel is not None:
        plt.xlabel(xlabel,size=15)
    if ylabel is not None:
        plt.ylabel(ylabel,size=15)
    if show:
        plt.show()

def pl_wordcloud(text_freq, title=None):
    """
    plot a word cloud
    :param text_freq: "wrd":freq
    :return:
    """
    wordcloud = WordCloud(background_color="white", width=800, height=400).generate_from_frequencies(text_freq)
    data = wordcloud.to_array()
    img = Image.fromarray(data, 'RGB')
    img = img.convert("RGBA")

    plt.imshow(img)
    plt.axis("on")
    plt.margins(x=0, y=0)
    if title is not None:
        plt.title(title)
    plt.show()

class WordPieceUtil(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize(self,word):
        return self.tokenizer.tokenize(word)

    def ids_to_tokens(self, nid):
        tokens = self.tokenizer.convert_ids_to_tokens(nid)
        return tokens

    def ids_to_token(self, nid):
        token = self.tokenizer.convert_ids_to_tokens([nid])[0]
        return token

    def tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def build_frequency(self):
        self.wp_file = "/home/hezq17/dataset/wikitext/wiki_text_wordpiece_nocase/tokens_bert_wiki_0.pickle"
        corpus = load_data(self.wp_file)
        counter = collections.Counter(corpus)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, counts = list(zip(*count_pairs))
        self.wp_list = np.array(words)
        self.counts = np.array(counts)/np.sum(counts)
        return self.wp_list, self.counts

class WikiBookDatasetBert(torch.utils.data.Dataset):
    def __init__(self, config):
        """
        WikiBookDatasetBert util for Bert.
        This util will dynamically add special tokens
        0: [PAD], 100: [UNK], 101: [CLS], 102: [SEP], 103: [MASK]
        """
        self.config(config)
        self.wiki_data_path = os.path.join(self.dataset_home,"wikitext/wiki_text_wordpiece_nocase")
        self.book_data_path = os.path.join(self.dataset_home,"bookcorpus/book_corpus_wordpiece_nocase")
        # self.wputil_path = os.path.join(self.dataset_home,"WordPieceUtil.data")
        # self.wputil = load_data(self.bpeutil_path)

        self.special_tokens = {"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102, "[MASK]": 103}
        # self.sentense_tokens = {".":1012, "!":999,"?":1029}
        self.sentense_tokens = [1012, 999, 1029]

        self.dataset = []
        print("Loading dataset ...")

        if self.mode=="train":
            for dataid in self.partition_wiki:
                data = load_data(os.path.join(self.wiki_data_path, "tokens_bert_wiki_%s.pickle" % dataid))
                self.dataset.append(data)
            for dataid in self.partition_book:
                data = load_data(os.path.join(self.book_data_path, "tokens_bert_book_%s.pickle" % dataid))
                self.dataset.append(data)
        elif self.mode=="val":
            for dataid in self.partition_wiki:
                data = load_data(os.path.join(self.wiki_data_path, "valid/tokens_bert_wiki_%s.pickle" % dataid))
                self.dataset.append(data)
            for dataid in self.partition_book:
                data = load_data(os.path.join(self.book_data_path, "valid/tokens_bert_book_%s.pickle" % dataid))
                self.dataset.append(data)
        else:
            raise Exception("Unknown mode")

        self.dataset = np.concatenate(self.dataset, axis=-1)
        print("Length of dataset, ", len(self.dataset))

        self.numblocks = int(len(self.dataset)/self.window)-10 # 10 block for safety margin
        self.idmap = np.array(range(self.numblocks))
        np.random.shuffle(self.idmap)

    def config(self, config):
        self.window = config.get("window",128)
        self.dataset_home = config.get("dataset_home","/home/hezq17/dataset/")
        self.max_data_len = config.get("max_data_len", 3000)
        self.partition_wiki = config.get("partition_wiki", [0,1])
        self.partition_book = config.get("partition_book", [0])
        self.mode = config.get("mode","train")

    def reshuffle(self):
        print("Reshuffle WikiBookDataset ...")
        np.random.shuffle(self.idmap)

    def get_dataloader(self,batch_size, num_workers=4, collate_fn=None):
        data_loader = torch.utils.data.DataLoader(
        self, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        return data_loader

    def find_sentence_start(self, ids):
        """
        Finding position of a start of a sentence
        :param ids:
        :return: startp
        """
        while self.dataset[ids] not in self.sentense_tokens:
            ids=ids+1
        return ids+1

    def find_num_sentence(self, startp, window):
        """
        Finding number of sentence within window, return a list
        :param startp:
        :param window:
        :return: [posi1, posi2, posi3 ...] where posi is sentence end mark.
        """
        sent_l = []
        idsent = startp
        ids = startp
        while idsent<startp+window-3: # 3 is for [cls] and [sep]*2
            if self.dataset[ids] not in self.sentense_tokens:
                ids=ids+1
            else:
                idsent=ids
                sent_l.append(ids)
                ids = ids + 1
        return sent_l

    def __getitem__(self, idx): ## sentence will
        """
        Get block idx
        :param idx:
        :return: [window, l_size]
        """
        idx = self.idmap[idx]
        startp = self.find_sentence_start(self.window * idx)
        sent_l = self.find_num_sentence(startp, self.window)
        assert len(sent_l)>=1

        ### Add [CLS] and [SEP] and [PAD]
        cls = self.special_tokens["[CLS]"]
        sep = self.special_tokens["[SEP]"]
        pad = self.special_tokens["[PAD]"]

        if len(sent_l)==1: ## Too long sentence
            ptdata = [cls]+list(self.dataset[startp:startp+self.window-1])
            assert len(ptdata) == self.window
        elif len(sent_l)==2: ## Not enough for two sentence, [sep] one sentence, and [pad] elsewhere
            ptdata = [cls] + list(self.dataset[startp:sent_l[0]+1]) + [sep]
            ptdata = ptdata + (self.window-len(ptdata))*[pad]
            assert len(ptdata) == self.window
        elif len(sent_l)>=3:
            midp = startp+int((sent_l[-2]-startp)/2) # the middle sentence closes to middle lenghth get [sep]
            picksent = np.argmin(np.abs(np.array(sent_l)-midp)[0:-2])
            pickp = sent_l[picksent]
            ptdata = [cls] + list(self.dataset[startp:pickp+1]) + [sep] + list(self.dataset[pickp+1:sent_l[-2]+1]) + [sep]
            ptdata = ptdata + (self.window - len(ptdata)) * [pad]
            assert len(ptdata) == self.window

        pt_word = torch.LongTensor(ptdata)
        return pt_word, pt_word

    def __len__(self):
        """
        number of text
        :return:
        """

        if self.max_data_len is None:
            numblocks = self.numblocks
        elif self.max_data_len<self.numblocks:
            numblocks = self.max_data_len
        else:
            numblocks = self.numblocks
        return int(numblocks)