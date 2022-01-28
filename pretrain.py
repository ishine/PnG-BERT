# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Pretrain transformer with Masked LM and Sentence Classification """

from random import randint, shuffle
from random import random as rand
import fire

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from dataloaders import tokenization
from models import modules
from bin import optimizer
from bin import train
from dataloaders import Pipeline, Preprocess4Pretrain, SentPairDataLoader
from utils.utils import set_seeds, get_device, get_random_word, truncate_tokens_pair
from models import BertModel4Pretrain

def main(train_cfg='configs/pretrain.json',
         model_cfg='configs/bert_base.json',
         data_file='/media/newhd/BookCorpus/books_large_p1.txt',
         model_file=None,
         data_parallel=True,
         vocab='/media/newhd/BookCorpus/vocab.txt',
         save_dir='./exp/bert/pretrain',
         log_dir='./exp/bert/pretrain/runs',
         max_len=512,
         max_pred=20,
         mask_prob=0.15):

    cfg = train.Config.from_json(train_cfg)
    model_cfg = modules.Config.from_json(model_cfg)

    set_seeds(cfg.seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    pipeline = [Preprocess4Pretrain(max_pred,
                                    mask_prob,
                                    list(tokenizer.vocab.keys()),
                                    tokenizer.convert_tokens_to_ids,
                                    max_len)]
    data_iter = SentPairDataLoader(data_file,
                                   cfg.batch_size,
                                   tokenize,
                                   max_len,
                                   pipeline=pipeline)

    model = BertModel4Pretrain(model_cfg)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()

    optim = optimizer.optim4GPU(cfg, model)
    trainer = train.Trainer(cfg, model, data_iter, optim, save_dir, get_device())

    writer = SummaryWriter(log_dir=log_dir) # for tensorboardX

    def get_loss(model, batch, global_step): # make sure loss is tensor
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

        logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
        loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids) # for masked LM
        loss_lm = (loss_lm*masked_weights.float()).mean()
        loss_clsf = criterion2(logits_clsf, is_next) # for sentence classification
        writer.add_scalars('data/scalar_group',
                           {'loss_lm': loss_lm.item(),
                            'loss_clsf': loss_clsf.item(),
                            'loss_total': (loss_lm + loss_clsf).item(),
                            'lr': optim.get_lr()[0],
                           },
                           global_step)
        return loss_lm + loss_clsf

    trainer.train(get_loss, model_file, None, data_parallel)


if __name__ == '__main__':
    fire.Fire(main)