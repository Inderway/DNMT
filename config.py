import torch

SRC_VOCAB_SIZE=32000
TGT_VOCAB_SIZE=32000
EMB_SIZE=512
NHEAD=8
FFN_HID_DIM=512
BATCH_SIZE=32
NUM_ENCODER_LAYERS=6
NUM_DECODER_LAYERS=6
NUM_EPOCHS=3
PAD_IDX=0
BOS_IDX=2
EOS_IDX=3
COLOR_LIST=['b', 'g', 'r', 'c', 'm', 'y']
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
early_stop = 5
lr = 3e-4
sentence_length=128

# greed decode的最大句子长度
MAX_LEN = 60
# beam size for bleu
beam_size = 3
# Label Smoothing
use_smoothing = False
# NoamOpt
use_noamopt = True

data_dir = './data'
train_data_path = './data/json/train.json'
dev_data_path = './data/json/dev.json'
test_data_path = './data/json/test.json'
model_path = './experiment/model.pth'
log_path = './experiment/train.log'
output_path = './experiment/output.txt'
image_path='./experiment/images'

# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]
gpu_id = '0'
device_id = [0,1,2,3,4,5,6,7]

# set device
if gpu_id != '':
    DEVICE = torch.device(f"cuda:1")
else:
    DEVICE = torch.device('cpu')
