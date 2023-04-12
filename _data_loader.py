import torch
import json
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import english_tokenizer_load
from utils import chinese_tokenizer_load

import _config
DEVICE = _config.device


def subsequent_mask(size):
    """
    Mask out subsequent positions.

    return:
        1 x size x size
    """
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)

    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    # k=1, elements below the first diagonal(the diagonal above the main diagonal) are zeros
    # triu applys to the last two dimensions
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    # elements reversed
    # [[[1, 0, 0,...,0],[1, 1, 0,..., 0],..., [1, 1, 1,..., 1]]]
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """
    Object for holding a batch of data with mask during training.

    Args:
        src_text(list): batch_size sentence string
        tgt_text(list): as above
        src(tensor): batch_size x max_len(of sequences in batch)
        tgt(tensor): as above
        pad(int)
    """
    def __init__(self, src_text, tgt_text, src, tgt=None, pad=0):
        self.src_text = src_text
        self.tgt_text = tgt_text
        src = src.to(DEVICE)
        self.src = src
        # 对于当前输入的句子非空部分进行判断成bool序列
        # batch_size x 1 x max_len
        # 1 corresponds to useful elements
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if tgt is not None:
            tgt = tgt.to(DEVICE)
            # decoder要用到的target输入部分
            # remove the last token of each sequence
            self.tgt = tgt[:, :-1]
            # decoder训练时应预测输出的target结果
            # remove the first token(bos) of each sequence
            self.tgt_y = tgt[:, 1:]
            # 将target输入部分进行attention mask
            # batch_size x max_len x max_len
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens = (self.tgt_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        
        return:
            mask(tensor): batch_size x max_len x max_len
        """
        # batch_size x 1 x max_len
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # tgt.size(-1)=max_len
        # batch_size x max_len x max_len
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class MTDataset(Dataset):
    def __init__(self, data_path):
        # sort to maka the length of sentences more balanced
        self.out_en_sent, self.out_cn_sent = self.get_dataset(data_path, sort=True)
        # load tokenization model
        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()
        self.PAD = self.sp_eng.pad_id()  # 0
        self.BOS = self.sp_eng.bos_id()  # 2
        self.EOS = self.sp_eng.eos_id()  # 3

    @staticmethod
    def len_argsort(seq):
        """
        传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标
        
        return:
            [ids of the sentences]
        """
        # seq: [sentence string], key is the length of seq[x]
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        """
        把中文和英文按照同样的顺序排序, 以英文句子长度排序的(句子下标)顺序为基准
        
        return:
            out_en_sent(list): English sentences
            out_cn_sent(list): Chinese sentences
        """
        dataset = json.load(open(data_path, 'r'))
        out_en_sent = []
        out_cn_sent = []
        # dataset: [[sentence_en_0, sentence_ch_0], [sentence_en_1, sentence_ch_1],...]
        
        # for testing
        if len(dataset)>1000:
            for idx, _ in enumerate(dataset[:1000]):
                out_en_sent.append(dataset[idx][0])
                out_cn_sent.append(dataset[idx][1])
        else:
            for idx, _ in enumerate(dataset):
                out_en_sent.append(dataset[idx][0])
                out_cn_sent.append(dataset[idx][1])


        if sort:
            sorted_index = self.len_argsort(out_en_sent)
            # array again
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_cn_sent = [out_cn_sent[i] for i in sorted_index]
        return out_en_sent, out_cn_sent

    def __getitem__(self, idx):
        '''
        return:
            [sentence_en, sentence_cn]
        '''
        eng_text = self.out_en_sent[idx]
        chn_text = self.out_cn_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):
        '''
        return:
            number of sentences
        '''
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        '''
        return:
            Batch
        '''
        # array sentences in batch in a list
        # src: en, tgt: cn
        # [sentence_en_0, sentence_en_1,...]
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]
        # [[id_sequence_0], [id_sequence_1],...] the lengths of each sequence are not equal 
        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        # make the token sequences have the same length
        # batch_size x longest_length 
        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD)
