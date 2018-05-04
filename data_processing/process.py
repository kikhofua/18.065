from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re

import torch
import torch.utils.data
from torch.autograd import Variable

SOS_token = 0  # Start of sentence
EOS_token = 1  # End of sentence

use_cuda = torch.cuda.is_available()


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.num_of_words = 2   # so far just SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_of_words
            self.word2count[word] = 1
            self.index2word[self.num_of_words] = word
            self.num_of_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):
    """
    Converts a Unicode string to plain ASCII because the spa-eng file is a Unicode file.
    Source: http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    """ Lowercase, trim, and remove non-letter characters """
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(lang1, lang2, reverse=False):
    """
    Read text file and split into lines, split lines into pairs and normalize
    :param lang1:
    :param lang2:
    :param reverse:
    :return: Lang object of normalized lang1, Lang object of normalized lang2, pairs
    """
    print("Reading lines...")

    # Read the file and split it into lines
    lines = open('../data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]  # contains two lists, one for each lang

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 10  # allows us to trim the dataset to include sentences no longer than MAX_LENGTH for speedy training
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(p):
    return len(p[0].split(" ")) < MAX_LENGTH and len(p[1].split(" ")) < MAX_LENGTH and p[1].startswith(eng_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.num_of_words)
    print(output_lang.name, output_lang.num_of_words)
    return input_lang, output_lang, pairs


# ----------------------------- NOW PREPARE FOR TRAINING -------------------------------- #

""" For each pair, we create an input tensor (indexes of the words in the input sentence) and 
    a target tensor (indexes of the words in the target sentence).
    We append EOS_token to both sequences."""


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variable_from_pair(pair, input_lang, output_lang):
    input_tensor = variable_from_sentence(input_lang, pair[0])
    target_tensor = variable_from_sentence(output_lang, pair[1])
    return input_tensor, target_tensor





