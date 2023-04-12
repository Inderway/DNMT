# to test code
# created by wei
# Feb 20, 2023

import os
import torch
from _data_loader import subsequent_mask
from torch.nn.functional import log_softmax
from __model import Embeddings, PositionalEncoding
import torch.nn as nn
import config
import matplotlib.pyplot as plt
device=config.DEVICE
from data_loader import MTDataset
from torch.utils.data import DataLoader

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    print(f'--generate_mask--\n{mask}\n, shape:{mask.shape}')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    print(f'--returned_mask--\n{mask}\n, shape:{mask.shape}')
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    print(f'--tgt_mask--\n{tgt_mask}\n, shape:{tgt_mask.shape}')
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    print(f'--src_mask--\n{src_mask}\n, shape:{src_mask.shape}')
    src_padding_mask = (src == config.PAD_IDX).transpose(0, 1)
    print(f'--src_padding_mask--\n{src_padding_mask}\n, shape:{src_padding_mask.shape}')
    tgt_padding_mask = (tgt == config.PAD_IDX).transpose(0, 1)
    print(f'--tgt_padding_mask--\n{tgt_padding_mask}\n, shape:{tgt_padding_mask.shape}')
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def plot(index, x, k, name_x, name_y):
    """
    plot the line chart

    Args:
        x(List): x axis
        k(List): line
        name_x: name of x axis
        name_y: name of y axis
    """
    plt.figure(index)
    plt.plot(x, k, 's-', color='r')
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    #plt.legend(loc='best')
    image_name=name_y+'.jpg'
    plt.savefig(os.path.join(config.image_path,image_name))

if __name__=='__main__':
    # 10x2
    src=torch.Tensor([[2,2],[4,21],[5,34],[52,57],[63,75],[3,91],[0,15],[0,19],[0,3],[0,0]])
    # 8x2
    tgt=torch.Tensor([[2,2],[25,16],[85,125],[81,8],[3,22],[0,3],[0,0],[0,0]])
    # torch.manual_seed(0)
    # train_dataset = MTDataset(config.train_data_path)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE,
    #                               collate_fn=train_dataset.collate_fn)
    # for i, batch in enumerate(train_dataloader):
    #     print (f'--------------Batch {i}--------------')
    #     for sent in batch.tgt_text:
    #         print(sent)

    plot(1, [1,2,3],[4,5,6],'x_1',"y_1")
    plot(2, [0,0.5,1],[1,2,3],'x_2',"y_2")
    