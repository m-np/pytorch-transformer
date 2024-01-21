# importing required libraries
import math
import copy
import time
import random
import spacy
import numpy as np
import os 

# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

# load and build datasets
import torchtext
from torchtext.data.functional import to_map_style_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import portalocker

import pandas as pd
from tqdm import tqdm


def load_tokenizers():
    """
    Load the German and English tokenizers provided by spaCy.

    Returns:
        spacy_de:     German tokenizer
        spacy_en:     English tokenizer
    """
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    print("Loaded English and German tokenizers.")
    return spacy_de, spacy_en


def tokenize(text: str, tokenizer):
    """
    Split a string into its tokens using the provided tokenizer.

    Args:
        text:         string 
        tokenizer:    tokenizer for the language
        
    Returns:
        tokenized list of strings       
    """
    return [tok.text.lower() for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index: int):
    """
    Return the tokens for the appropriate language.

    Args:
        data_iter:    text here 
        tokenizer:    tokenizer for the language
        index:        index of the language in the tuple | (de=0, en=1)
        
    Yields:
        sequences based on index       
    """
    for from_tuple in data_iter:
        yield tokenizer(from_tuple[index])


def build_vocabulary(
                    spacy_de, 
                    spacy_en, 
                    train_iter, 
                    val_iter, 
                    test_iter, 
                    min_freq: int = 2):
  
    def tokenize_de(text: str):
        """
          Call the German tokenizer.

          Args:
              text:         string 
              min_freq:     minimum frequency needed to include a word in the vocabulary

          Returns:
              tokenized list of strings       
        """
        return tokenize(text, spacy_de)
    
    def tokenize_en(text: str):
        """
          Call the English tokenizer.

          Args:
              text:         string 

          Returns:
              tokenized list of strings       
        """
        return tokenize(text, spacy_en)

    print("Building German Vocabulary...")

    train = train_iter
    val = val_iter
    test = test_iter

    # generate source vocabulary
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0), # tokens for each German sentence (index 0)
        min_freq=min_freq, 
        specials=["<bos>", "<eos>", "<pad>", "<unk>"],
    )

    print("Building English Vocabulary...")

    # generate target vocabulary
    vocab_trg = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1), # tokens for each English sentence (index 1)
        min_freq=2, # 
        specials=["<bos>", "<eos>", "<pad>", "<unk>"],
    )

    # set default token for out-of-vocabulary words (OOV)
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_trg.set_default_index(vocab_trg["<unk>"])

    return vocab_src, vocab_trg


def load_vocab(
        spacy_de, 
        spacy_en, 
        train_iter, 
        val_iter, 
        test_iter, 
        min_freq: int = 2):
    """
    Args:
        spacy_de:     German tokenizer
        spacy_en:     English tokenizer
        min_freq:     minimum frequency needed to include 
                      a word in the vocabulary

    Returns:
        vocab_src:    German vocabulary
        vocab_trg:     English vocabulary       
    """
    if not os.path.exists("vocab.pt"):
        # build the German/English vocabulary if it does not exist
        vocab_src, vocab_trg = build_vocabulary(spacy_de, 
                                                spacy_en, 
                                                train_iter, 
                                                val_iter, 
                                                test_iter, 
                                                min_freq)
        # save it to a file
        torch.save((vocab_src, vocab_trg), "vocab.pt")
    else:
        # load the vocab if it exists
        vocab_src, vocab_trg = torch.load("vocab.pt")

    print("Finished.\nVocabulary sizes:")
    print("\tSource:", len(vocab_src))
    print("\tTarget:", len(vocab_trg))
    return vocab_src, vocab_trg


def data_process(
        raw_data, 
        spacy_de, 
        spacy_en, 
        vocab_src, 
        vocab_trg):
    """
    Process raw sentences by tokenizing and converting to integers based on 
    the vocabulary.

    Args:
        raw_data:     German-English sentence pairs 
        spacy_de:     German Tokenizer
        spacy_en:     English Tokenizer
        vocab_src:    Source vocabulary
        vocab_trg:    Target vocabulary
    Returns:
        data:         tokenized data converted to index based on vocabulary   
    """
    data = []
    # loop through each sentence pair
    for (raw_de, raw_en) in tqdm(raw_data):
        de_tensor_ = []
        # tokenize the sentence and convert each word to an integers
        for token in spacy_de.tokenizer(raw_de):
            de_tensor_.append(vocab_src[token.text.lower()])
            
        en_tensor_ = []
        # tokenize the sentence and convert each word to an integers
        for token in spacy_en.tokenizer(raw_en):
            en_tensor_.append(vocab_trg[token.text.lower()])
            
        de_tensor_ = torch.tensor(de_tensor_, dtype=torch.long)
        en_tensor_ = torch.tensor(en_tensor_, dtype=torch.long)
        # append tensor representations
        data.append((de_tensor_, en_tensor_))
    return data


def generate_batch(
        data_batch,
        ):
    """
    Process indexed-sequences by adding <bos>, <eos>, and <pad> tokens.

    Args:
        data_batch:     German-English indexed-sentence pairs

    Returns:
        two batches:    one for German and one for English
    """
    de_batch, en_batch = [], []

    # for each sentence
    for (de_item, en_item) in data_batch:
        # add <bos> and <eos> indices before and after the sentence
        de_temp = torch.cat([torch.tensor([BOS_IDX]), 
                             de_item, 
                             torch.tensor([EOS_IDX])], dim=0).to(device)
        en_temp = torch.cat([torch.tensor([BOS_IDX]), 
                             en_item, 
                             torch.tensor([EOS_IDX])], dim=0).to(device)

        # add padding
        de_batch.append(pad(de_temp,(0, # dimension to pad
                                MAX_PADDING - len(de_temp), # amount of padding to add
                              ),value=PAD_IDX,))

        # add padding
        en_batch.append(pad(en_temp,(0, # dimension to pad
                                MAX_PADDING - len(en_temp), # amount of padding to add
                              ),
                              value=PAD_IDX,))

    return torch.stack(de_batch), torch.stack(en_batch)



