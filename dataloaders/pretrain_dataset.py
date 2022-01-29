from random import randint, shuffle
from random import random as rand
import torch
import torch.nn as nn
from utils.utils import set_seeds, get_device, get_random_word, truncate_tokens_pair
from nemo_text_processing.text_normalization.normalize import Normalizer
from g2p_en import G2p
import string
import numpy as np

def seek_random_offset(f, back_margin=2000):
    """ seek random offset of file pointer """
    f.seek(0, 2)
    # we remain some amount of text to read
    max_offset = f.tell() - back_margin
    f.seek(randint(0, max_offset), 0)
    f.readline() # throw away an incomplete sentence




class SentPairDataLoader():
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, file, batch_size, tokenize, tokenizer, max_len, short_sampling_prob=0.1, pipeline=[]):
        super().__init__()
        self.f_pos = open(file, "r", encoding='utf-8', errors='ignore') # for a positive sample
        self.f_neg = open(file, "r", encoding='utf-8', errors='ignore') # for a negative (random) sample
        self.tokenize = tokenize # tokenize function
        self.tokenizer = tokenizer
        self.max_len = max_len # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.normalizer = Normalizer(input_case='lower_cased', lang='en')
        self.g2p  = G2p()
        self.phonemes = self.g2p.phonemes
        

    def read_tokens(self, f, length, discard_last_and_restart=True):
        """ Read tokens from file pointer with limited length """
        tokens = []
        phn_tokens = []
        tokenized_word_ids = []
        tokenized_phn_ids = []
        tokens_word_ids = []
        phn_word_ids = []
        word_count=0
        phn_word_count=0
        while len(tokens) < length:
            line = f.readline()
            if not line: # end of file
                return None
            if not line.strip(): # blank line (delimiter of documents)
                if discard_last_and_restart:
                    tokens = [] # throw all and restart
                    continue
                else:
                    return tokens # return last tokens in the document
            

            no_punc_string = line.rstrip('\n').translate(str.maketrans('', '', string.punctuation))
            clean_string = " ".join(list(filter(None, no_punc_string.split(" "))))
            normalize_text = self.normalizer.normalize(clean_string, verbose=False)
            
            for word in normalize_text.split(' '):
                word_tokens =  self.tokenize(word)
                for i in range(len(word_tokens)):
                    tokens_word_ids.append(word_count)
                    tokens.append(word_tokens[i])
                    tokenized_word_ids.append(self.tokenizer.convert_tokens_to_ids([word_tokens[i]])[0])

                word_count+=1


            for word in normalize_text.split(' '):
                phns_seq = self.g2p(word)
                for i in range(len(phns_seq)):
                    try:
                        tokenized_phn_ids.append(self.g2p.p2idx[phns_seq[i]])
                    except:
                        continue
                    phn_word_ids.append(phn_word_count)
                    phn_tokens.append(phns_seq[i])
                    
                phn_word_count+=1

        return tokens, phn_tokens, tokens_word_ids, phn_word_ids, tokenized_word_ids,tokenized_phn_ids

    def __iter__(self): # iterator to load data
        while True:
            batch = []
            for i in range(self.batch_size):
                # sampling length of each tokens_a and tokens_b
                # sometimes sample a short sentence to match between train and test sequences
                len_tokens = randint(1, int(self.max_len / 2)) \
                    if rand() < self.short_sampling_prob \
                    else int(self.max_len / 2)

                tokens_a, tokens_b, tokens_word_ids, phn_word_ids, tokenized_word_ids,tokenized_phn_ids = self.read_tokens(self.f_pos, len_tokens, True)
                
                instance = (tokenized_word_ids, tokenized_phn_ids,tokens_word_ids, phn_word_ids)
                for proc in self.pipeline:
                    instance = proc(instance)
                batch.append(instance)

            # To Tensor
            batch_tensors = batch
            batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
            yield batch_tensors
        

def batch_fit(tokens_a, tokens_b, tokens_word_ids, phn_word_ids, max_len):
    assert len(set(tokens_word_ids)) == len(set(phn_word_ids))
    unique_word_ids = list(set(tokens_word_ids))
    while True:
        total_len = len(tokens_a) + len(tokens_b)
        if total_len > max_len:
            word_to_pop = unique_word_ids.pop()
            num_tokens_remove = tokens_word_ids.count(word_to_pop)
            for i in range(num_tokens_remove):
                tokens_a.pop()
                tokens_word_ids.pop()

            num_tokens_remove = phn_word_ids.count(word_to_pop)
            for i in range(num_tokens_remove):
                tokens_b.pop()
                phn_word_ids.pop()

        else:
            break
        

class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Pretrain(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512):
        super().__init__()
        self.max_pred = max_pred # max tokens of prediction
        self.mask_prob = mask_prob # masking probability
        self.vocab_words = vocab_words # vocabulary (sub)words
        self.indexer = indexer # function from token to token index
        self.max_len = max_len

    def __call__(self, instance):
        tokens_a, tokens_b, tokens_word_ids, phn_word_ids  = instance
        batch_fit(tokens_a, tokens_b, tokens_word_ids, phn_word_ids,self.max_len - 3 )
        # -3  for special tokens [CLS], [SEP], [SEP]
        #truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)
        # Add Special Tokens
        tokens = self.indexer(['[CLS]']) + tokens_a + self.indexer(['[SEP]']) + tokens_b + self.indexer(['[SEP]'])
        word_ids = [0]+list(np.array(tokens_word_ids)+1)+[0]+list(np.array(phn_word_ids)+1)+[0]
        segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)
        input_mask = [1]*len(tokens)
        
        # For masked Language Models
        masked_tokens, masked_pos = [], []
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(tokens)*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = [i for i, token in enumerate(tokens)
                    if token != self.indexer(['[CLS]'])[0] and token != self.indexer(['[SEP]'])[0]]
        
        shuffle(cand_pos)
        for pos in cand_pos[:n_pred]:
            masked_tokens.append(tokens[pos])
            masked_pos.append(pos)
            if rand() < 0.8: # 80%
                tokens[pos] = self.indexer(['[MASK]'])[0]
            elif rand() < 0.5: # 10%
                tokens[pos] = self.indexer([get_random_word(self.vocab_words)])[0]
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = tokens
        masked_ids = masked_tokens
        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        input_mask.extend([0] * n_pad)    
        word_ids.extend([0] * n_pad)
        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0]*n_pad)
            masked_pos.extend([0]*n_pad)
            masked_weights.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights,word_ids)
        


if __name__=='__main__':
    vocab ='/media/newhd/BookCorpus/vocab.txt'
    data_file ='/media/newhd/BookCorpus/books_large_p1.txt'

    from dataloaders import tokenization
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))
    max_len=512
    max_pred=20
    mask_prob=0.15
    pipeline = [Preprocess4Pretrain(max_pred,
                                    mask_prob,
                                    list(tokenizer.vocab.keys()),
                                    tokenizer.convert_tokens_to_ids,
                                    max_len)]
    data_iter = SentPairDataLoader(data_file,
                                   5,
                                   tokenize,
                                   tokenizer,
                                   max_len,
                                   pipeline=pipeline)
    