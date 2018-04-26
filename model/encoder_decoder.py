import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 10  # length of longest sentence in processed data


class EncoderRNN(nn.Module):
    """
    This is an RNN that outputs some value for every word from the input sentence.
    Thus for every input word, the encoder outputs a vector and a hidden state.
    It uses the hidden state for the next input word.
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        # https://stackoverflow.com/questions/42479902/how-view-method-works-for-tensor-in-torch?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttentionDecoderRNN(nn.Module):
    """
    We use an Attention in order to allow the Decoder network to "focus" on a different part of the encoder's outputs
    for every step of the decoder's own outputs.

    First we calculate the attention weights with a feed-forward layer (attn) using the decoder's input and hidden state
    as inputs. To create and train this layer we have to choose a maximum sentence length because the sentences in the
    data all have different lengths.

    Then we multiply the attention weigts by the encoder output vectors to create a weighted combination. The result
    (attn_applied) should contain information about that specific part of the input sequence and help the decoder choose
    the right output words

    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # http://pytorch.org/docs/master/torch.html#torch.bmm
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



