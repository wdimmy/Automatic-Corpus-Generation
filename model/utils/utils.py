#-*- coding:utf-8 -*-
import torch
import torch.utils.data as data
from torch.autograd import Variable
import logging
import os
import codecs
import random
from bs4 import BeautifulSoup

UNK_token=0
PAD_token=1
EOS_token=2
SOS_token=3

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False

class Lang:
    def __init__(self):
        self.word2index = {}
        self.tag2index = {}
        self.word2count = {}
        self.tag2count = {}
        self.index2word = {UNK_token: 'UNK', PAD_token: "PAD", EOS_token: "EOS", SOS_token: "SOS"}
        self.index2tag = {}
        self.n_words = 4  # Count default tokens
        self.n_tags = 0  # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_tags(self, sentence):
        for word in sentence.split(' '):
            self.index_tag(word)


    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def index_tag(self, word):
        if word not in self.tag2index:
            self.tag2index[word] = self.n_tags
            self.tag2count[word] = 1
            self.index2tag[self.n_tags] = word
            self.n_tags += 1
        else:
            self.tag2count[word] += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, src_seq, trg_seq, src_word2id, trg_word2id):
        """Reads source and target sequences from txt files."""
        self.src_seqs = src_seq
        self.trg_seqs = trg_seq
        self.num_total_seqs = len(self.src_seqs)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        src_seq = self.preprocess(src_seq, self.src_word2id, trg=False)  # change to index
        trg_seq = self.preprocess(trg_seq, self.trg_word2id)  #
        return src_seq, trg_seq, self.src_seqs[index], self.trg_seqs[index]

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if (trg):
            sequence = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')]
            sequence = torch.Tensor(sequence)
        else:
            sequence = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')]
            sequence = torch.Tensor(sequence)
        return sequence


def collate_fn(data):
    def merge(sequences, max_len):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)
    # seperate source and target sequences
    src_seqs, trg_seqs, src_plain, trg_plain = zip(*data)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs, None)
    trg_seqs, trg_lengths = merge(trg_seqs, None)

    src_seqs = Variable(src_seqs).transpose(0, 1)
    trg_seqs = Variable(trg_seqs).transpose(0, 1)
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
    return src_seqs, src_lengths, trg_seqs, trg_lengths,  src_plain, trg_plain

def read_langs(file_name):
    logging.info(("Reading lines from {}".format(file_name)))
    total_data=[]
    with codecs.open(file_name,"r", encoding="utf-8") as fin:
        data = ""
        for line in fin:
            if line.strip():
                data += line
            else:
                locations = []
                soup = BeautifulSoup(data, "html.parser")
                for sentence in soup.find_all("sentence"):
                    text = sentence.find("text").text.strip()
                    mistakes = sentence.find_all("mistake")
                    for mistake in mistakes:
                        location = mistake.find("location").text.strip()
                        locations.append(int(location))
                sen =  list(text)
                tags = ["0" for _ in range(len(sen))]

                for i in locations:
                    tags[i-1] = "1"
                    total_data.append([" ".join(sen), " ".join(tags)])
                data = ""
    return total_data

def get_seq(pairs,lang,batch_size,type):
    x_seq = []
    y_seq = []
    for pair in pairs:
        x_seq.append(pair[0])
        y_seq.append(pair[1])
        if(type):
            lang.index_words(pair[0])
            lang.index_tags(pair[1])

    dataset = Dataset(x_seq, y_seq,lang.word2index, lang.tag2index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              collate_fn=collate_fn)
    return data_loader


def prepare_data_seq(filename, lang, isSplit=False, batch_size=64):
    if isSplit:
        data_path = filename
        data = read_langs(data_path)

        max_train_len = max([len(d[0].split(' ')) for d in data])
        logging.info("Number: {} and  Maxlen: {}".format(len(data), max_train_len))

        data = get_seq(data, lang, batch_size, True)


        logging.info("Vocab_size %s " % lang.n_words)
        logging.info("USE_CUDA={}".format(USE_CUDA))
        return data

    else:
        data_path = filename
        total_data = read_langs(data_path)

        random.shuffle(total_data)

        train = total_data[:int(len(total_data)*0.9)]
        dev = total_data[int(len(total_data)*0.9):]
        #test = total_data[int(len(total_data) * 0.9):]

        max_train_len = max([len(d[0].split(' ')) for d in train])
        max_dev_len = max([len(d[0].split(' ')) for d in dev])
        #max_test_len = max([len(d[0].split(' ')) for d in test])

        logging.info("Train: Number: {} and  Maxlen: {}".format(len(train), max_train_len))
        logging.info("Dev: Number: {} and  Maxlen: {}".format(len(dev), max_dev_len))
        #logging.info("Test: Number: {} and  Maxlen: {}".format(len(test),max_test_len))

        train = get_seq(train, lang, batch_size, True)
        dev = get_seq(dev, lang, batch_size, True)
        #test = get_seq(test, lang, batch_size, True)

        logging.info("Vocab_size %s " % lang.n_words)
        logging.info("USE_CUDA={}".format(USE_CUDA))

        return train, dev
