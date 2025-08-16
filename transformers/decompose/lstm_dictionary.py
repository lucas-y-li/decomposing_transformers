import os
import torch
from collections import defaultdict


class Dictionary(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []

        vocab = open(path, encoding="utf8").read()
        self.word2idx = {w: i for i, w in enumerate(vocab.split())}
        self.idx2word = [w for w in vocab.split()]
        self.vocab_file_exists = True

    def add_word(self, word):
        self.word2freq[word] += 1
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)

    def create_vocab(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.add_word(word)

    def tokenize_file(self, path):
        assert os.path.exists(path)

        with open(path, 'r', encoding="utf8") as f:
            ids = []
            for line in f:
                idx = []
                words = line.split()
                for word in words:
                    if word in self.word2idx:
                        idx.append(self.word2idx[word])
                    else:
                        idx.append(self.word2idx["<unk>"])
                idx = torch.LongTensor(idx)
                ids.append(idx)

        return ids

    def tokenize(self, text, add_eos=False):
        words = text.split()
        idx = []
        for word in words:
            if word in self.word2idx:
                idx.append(self.word2idx[word])
            else:
                idx.append(self.word2idx["<unk>"])
        if add_eos:
            idx.append(self.word2idx["<eos>"])

        idx = torch.LongTensor(idx)

        return idx
