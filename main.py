# main
# created by wei
# April 7, 2023

import config
from model import Transformer
import torch.nn as nn
import torch
import utils
import logging
from data_loader import MTDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from timeit import default_timer as timer
from data_loader import generate_square_subsequent_mask
from utils import english_tokenizer_load, chinese_tokenizer_load
import sacrebleu
import matplotlib.pyplot as plt

device=config.DEVICE
torch.manual_seed(0)

def run(model):
    utils.set_logger(config.log_path)

    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path) 
    test_dataset = MTDataset(config.test_data_path)

    logging.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
                                 collate_fn=test_dataset.collate_fn)

    logging.info("-------- Get Dataloader! --------")
    # initialization
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    loss_fn=nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
    optimizer=torch.optim.Adam(
        model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
    num_epochs=config.NUM_EPOCHS
    train_losses=[]
    val_losses=[]
    sacreBLEUs=[]
    epochs=[i for i in range(1,num_epochs+1)]
    for epoch in range(1, num_epochs+1):
        start_time=timer()
        train_loss=train_epoch(model, train_dataloader, optimizer, loss_fn)
        train_losses.append(train_loss)
        val_loss=evaluate(model,dev_dataloader, loss_fn)
        val_losses.append(val_loss)
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')
        sacrebleu_score=test(model, test_dataloader)
        sacreBLEUs.append(sacrebleu_score)
        end_time=timer()
        print(f'Test bleu score = {sacrebleu_score}, Epoch Time: {((end_time-start_time)/60):.2f}min')
        model.load_state_dict(torch.load(config.model_path))
    plot(1,epochs, train_losses, 'epoch', 'train loss')
    plot(2,epochs, val_losses, 'epoch', 'val loss')
    plot(3,epochs, sacreBLEUs, 'epoch', 'sacreBLEU')
    print(f'train_losses:\n{train_losses}\nval_losses:\n{val_losses}\nsacreBLEUs:\n{sacreBLEUs}')
    
    

def train_epoch(model,data,optimizer, loss_fn):
    model.train()
    losses=0
    
    for batch in tqdm(data):
        # batch_size x sent_len 
        src=batch.src
        #print(batch.src_text)
        tgt_input=batch.tgt_input
        tgt_out=batch.tgt_out
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask=\
            (batch.src_mask,batch.tgt_mask,batch.src_padding_mask, batch.tgt_padding_mask)
        # src: batch_size x len x emb
        logits=model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()
        loss=loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses+=loss.item()
    return losses / len(list(data))

def evaluate(model,data, loss_fn):
    model.eval()
    losses=0
    loss_min=20.0
    for batch in tqdm(data):
        src=batch.src
        tgt_input=batch.tgt_input
        tgt_out=batch.tgt_out
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask=\
            (batch.src_mask,batch.tgt_mask,batch.src_padding_mask, batch.tgt_padding_mask)
        # src: batch_size x len x emb
        logits=model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        loss=loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        if loss<loss_min:
            loss_min=loss
            torch.save(model.state_dict(), config.model_path)
        losses+=loss.item()
    return losses/len(list(data))

def test(model, data):
    tgt=[]
    res=[]
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        for batch in tqdm(data):
            tgt_text=batch.tgt_text
            src=batch.src
            num_tokens=src.size(-1)
            src_mask=(torch.zeros(num_tokens, num_tokens)).type(torch.bool)
            translations=[]
            for i, ele in enumerate(src):
                src_sent=ele.unsqueeze(0)
                tgt_tokens=greedy_decode(model, src_sent,src_mask,max_len=num_tokens+5, start_symbol=config.BOS_IDX)
                sp_zh=chinese_tokenizer_load()
                translation=[sp_zh.decode_ids(_s) for _s in tgt_tokens]
                sent=''.join(translation)
                translations.append(sent)
            tgt.extend(tgt_text)
            res.extend(translations)
    tgt=[tgt]
    bleu=sacrebleu.corpus_bleu(res,tgt,tokenize='zh')
    return float(bleu.score)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src=src.to(device)
    src_mask=src_mask.to(device)
    memory=model.encode(src, src_mask)
    ys=torch.ones(1,1).fill_(start_symbol).type(torch.long).to(device)
    result=[]
    for i in range(max_len-1):
        memory=memory.to(device)
        tgt_mask=(generate_square_subsequent_mask(ys.size(1)).type(torch.bool)).to(device)
        out=model.decode(ys, memory, tgt_mask)
        prob=model.generator(out[:,-1])
        _, next_word=torch.max(prob, dim=1)
        next_word=next_word.item()
        ys=torch.cat([ys, torch.ones(1,1).type_as(src.data).fill_(next_word)],dim=1)
        if next_word==config.EOS_IDX:
            break
    for ele in ys[0]:
        result.append(ele.item())
    return result

def translate(model, src_sentence):
    model.eval()
    src= [[config.BOS_IDX] + english_tokenizer_load().EncodeAsIds(src_sentence) + [config.EOS_IDX]]
    input = torch.LongTensor(src)
    num_tokens=len(input[0])
    src_mask=(torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens=greedy_decode(model, input, src_mask, max_len=num_tokens+5, start_symbol=config.BOS_IDX)
    sp_zh=chinese_tokenizer_load()
    result=[sp_zh.decode_ids(_s) for _s in tgt_tokens]
    return ''.join(result)

def plot(index, x, k, name_x, name_y):
    """
    plot the line chart

    Args:
        index(Int): index of the image
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



if __name__ == "__main__":
    import os
    model=Transformer(config.NUM_ENCODER_LAYERS, config.NUM_DECODER_LAYERS, config.EMB_SIZE, config.NHEAD, config.SRC_VOCAB_SIZE, config.TGT_VOCAB_SIZE, config.FFN_HID_DIM).to(device)
    # if os.path.exists(config.model_path):
    #     logging.info("-------- Model exists! --------")
    #     model.load_state_dict(torch.load(config.model_path))
    # else:
    run(model)
    # test_dataset = MTDataset(config.test_data_path)
    # test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
    #                              collate_fn=test_dataset.collate_fn)
    # test(model, test_dataloader)
    print(translate(model,"Hello World!"))
