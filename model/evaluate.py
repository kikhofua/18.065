from model.train import *
from data_processing.process import *

import random


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_tensor = tensor_from_sentence(input_lang, sentence)
    input_length = input_tensor.size()[0]
    encoder_hidden = encoder.init_hidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # initialize with SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di+1]


def evaluate_randomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == "__main__":
    input_lang, output_lang, pairs = prepare_data('eng', 'spa', True)

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.num_of_words,  hidden_size)
    attn_decoder1 = AttentionDecoderRNN(hidden_size, output_lang.num_of_words, dropout_p=0.1)

    train_iters(pairs, input_lang, output_lang, encoder1, attn_decoder1, 75000, print_every=5000)
    evaluate_randomly(encoder1, attn_decoder1)

