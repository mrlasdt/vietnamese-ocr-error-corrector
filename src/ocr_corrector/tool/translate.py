import torch
import numpy as np
from torch.nn.functional import log_softmax, softmax
# from icecream import ic
from .vocab import Vocab
from .common import alphabet, THRESH
vocab_test = Vocab(alphabet)


def translate(src_text, model, device, max_seq_length=256, sos_token=1, eos_token=2):
    """
    # data: BxCxHxW
    src_text: tensor of sentence NxL = number of sentences x sentences maxlen
            + encoder of seq2seq = nn.Embedding: map integer (1-229 = vocab_size) to vector 1x256 (1x hidden dim)

    translated_sentence: np array of shape Nx min(L, max_seq_length)
    """
    model.eval()
    # ic(src_text.shape)
    src_text = src_text.to(device)
    with torch.no_grad():
        # memory = encoder_output NxLxhidden_dim*2, hidden_state num_directionsxLxhidden_dim (num_direction = 2 if bidirectional=true)
        memory = model.forward_encoder(src_text)
        translated_sentence = [[sos_token] * len(src_text)]  # init tgt_input by <start> token
        max_length = 0
        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = torch.LongTensor(np.array(translated_sentence)).to(device)
            # output Nx1xvocab_size, memory same as encoder_memory
            output, memory = model.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)  # prob distribution of each character in vocabulary

            # ### BACK UP ### use to train
            output = output.to('cpu')
            values, indices = torch.topk(output, 5)
            # select top 5 prob characters, indices Nx1x5 = integer = chararacter code (1-229) in vocab
            indices = indices[:, -1, 0].tolist()  # only select top1 -> greedy ->maybe improve by beamsearch
            translated_sentence.append(indices)
            ### END BACK UP ###

            # ### use to infer ###
            # values, indices = torch.topk(output, 1)  # greedy
            # values = values[:, -1, 0].tolist()
            # indices = indices[:, -1, 0].tolist()
            # threshed_indices = []
            # next_src_text = src_text[:, max_length + 1].tolist() if max_length < src_text.shape[1] else [
            #     [eos_token] * len(src_text)]     # len(src_text) = src_text.shape[0]
            # for i, index in enumerate(indices):
            #     if values[i] >= THRESH:
            #         threshed_indices.append(index)
            #     else:
            #         threshed_indices.append(next_src_text[i])
            # translated_sentence.append(threshed_indices)
            # ### end infer ###
            max_length += 1
            del output
        translated_sentence = np.asarray(translated_sentence).T
    return translated_sentence
