# to test code
# created by wei
# Feb 20, 2023

import os
import torch
from data_loader import subsequent_mask

pad=0
tgt=[[1,2,3,4,5,0,0],[6,7,8,9,0,0,0]]
tgt=torch.tensor(tgt)
# tgt_mask = (tgt != pad).unsqueeze(-2)
# print("tgt_mask:")
# print(tgt_mask)
# print(tgt_mask.shape)
# latter=subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
# print("latter:")
# print(latter)
# print(latter.shape)
# print("and:")
# res=tgt_mask&latter
# print(res)
# print(res.shape)
print((tgt!= pad).data.sum())
