import utils
import _config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn import Transformer as tfm
from _train import train, test, translate
from _data_loader import MTDataset
from utils import english_tokenizer_load
from _model import make_model, LabelSmoothing


class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """for batch_size 32, 5530 steps for one epoch, 2 epoch for warm-up"""
    return NoamOpt(model.src_embed[0].d_model, 1, 10000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def run():
    utils.set_logger(_config.log_path)

    train_dataset = MTDataset(_config.train_data_path)
    dev_dataset = MTDataset(_config.dev_data_path) 
    test_dataset = MTDataset(_config.test_data_path)

    logging.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=_config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=_config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=_config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    logging.info("-------- Get Dataloader! --------")
    # 初始化模型
    model = make_model(_config.src_vocab_size, _config.tgt_vocab_size, _config.n_layers,
                       _config.d_model, _config.d_ff, _config.n_heads, _config.dropout)

    logging.info("-------- Make Model! --------")
    model_par = torch.nn.DataParallel(model)
    # 训练
    if _config.use_smoothing:
        criterion = LabelSmoothing(size=_config.tgt_vocab_size, padding_idx=_config.padding_idx, smoothing=0.1)
        criterion.cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    if _config.use_noamopt:
        optimizer = get_std_opt(model)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=_config.lr)
    logging.info("-------- Train! --------")    
    train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer)
    logging.info("-------- Test! --------")
    test(test_dataloader, model, criterion)


def check_opt():
    """check learning rate changes"""
    import numpy as np
    import matplotlib.pyplot as plt
    model = make_model(_config.src_vocab_size, _config.tgt_vocab_size, _config.n_layers,
                       _config.d_model, _config.d_ff, _config.n_heads, _config.dropout)
    opt = get_std_opt(model)
    # Three settings of the lrate hyperparameters.
    opts = [opt,
            NoamOpt(512, 1, 20000, None),
            NoamOpt(256, 1, 10000, None)]
    plt.plot(np.arange(1, 50000), [[opt.rate(i) for opt in opts] for i in range(1, 50000)])
    plt.legend(["512:10000", "512:20000", "256:10000"])
    plt.show()


def one_sentence_translate(sentence, beam_search=True):
    """
    Args:
        sentence(string): source sentence
    """
    # 初始化模型
    model = make_model(_config.src_vocab_size, _config.tgt_vocab_size, _config.n_layers,
                       _config.d_model, _config.d_ff, _config.n_heads, _config.dropout)
    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3
    # Encode the sentence to id sequence, then add begin symbol and end symbol
    # [[int, int, int,..., int]]
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sentence) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(_config.device)
    translate(batch_input, model, use_beam=beam_search)


def translate_example():
    """单句翻译示例"""
    sent = "The near-term policy remedies are clear: raise the minimum wage to a level that will keep a " \
           "fully employed worker and his or her family out of poverty, and extend the earned-income tax credit " \
           "to childless workers."
    # tgt: 近期的政策对策很明确：把最低工资提升到足以一个全职工人及其家庭免于贫困的水平，扩大对无子女劳动者的工资所得税减免。
    one_sentence_translate(sent, beam_search=True)


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
    import warnings
    warnings.filterwarnings('ignore')
    run()
    translate_example()
