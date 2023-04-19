import torch
import numpy as np
import re
from collections import defaultdict
from .model.seq2seq import Seq2Seq
from .model.transformer import LanguageTransformer
from .tool.vocab import Vocab
from .tool.common import alphabet, no_space_char
from .tool.translate import translate
from .tool.utils import get_bucket


class Predictor(object):
    def __init__(self, device, model_type='seq2seq', weight_path='./weights/seq2seq_0.pth'):
        if model_type == 'seq2seq':
            self.model = Seq2Seq(len(alphabet), encoder_hidden=256, decoder_hidden=256)
        elif model_type == 'transformer':
            self.model = LanguageTransformer(
                len(alphabet),
                d_model=210, nhead=6, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=768,
                max_seq_length=256, pos_dropout=0.1, trans_dropout=0.1)
        self.device = device
        self.model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        self.model = self.model.to(device)
        self.vocab = Vocab(alphabet)

    def process_custom(self, paragraph, NGRAM):  # custom by hungbnt
        phrases = self.extract_phrase(paragraph)
        inputs = []
        masks = []
        # group by n-grams
        for phrase in phrases:
            words = phrase.split()
            if len(words) < 2 or not re.match("\w[\w ]+", phrase):
                inputs.append(phrase)
                masks.append(False)
            else:
                for i in range(0, len(words), NGRAM):
                    if len(words) - i < NGRAM:  # last words
                        inputs.append(' '.join(words[i:]))
                        if len(words) - i < 2 or not re.match(
                                "\w[\w ]+", ' '.join(words[i:])):  # skip single words leftover
                            masks.append(False)
                            continue
                    else:
                        inputs.append(' '.join(words[i:i + NGRAM]))
                    masks.append(True)
        return inputs, masks

    # @staticmethod
    # def _process_overlap(NGRAM, inputs, masks):
    #     masks_overlap = []
    #     inputs_overlap = []
    #     pivot = int(NGRAM / 2)
    #     for i, mask in enumerate(masks[:-1]):
    #         if mask and masks[i + 1]:
    #             masks_overlap.append(True)
    #             inputs_overlap.append(' '.join(inputs[i].split()[
    #                                   pivot:] + inputs[i + 1].split()[:pivot]))
    #         else:
    #             masks_overlap.append(False)
    #             inputs_overlap.append('')
    #     masks_overlap.append(False)
    #     inputs_overlap.append('')
    #     # n = len(inputs[-1].split())
    #     # if n < NGRAM and masks[-1] and masks[-2]:
    #     #     inputs_overlap.append(' '.join(inputs[-2].split()[NGRAM - n + 1:] + inputs[-1].split()))
    #     #     masks_overlap.append(True)
    #     return inputs_overlap, masks_overlap, masks

    # def process_overlap(self, paragraph, NGRAM):
    #     '''
    #     process with overlap option
    #     overlap = False: similar to process
    #     overlap = True: skip first NGRAM/2 characters
    #     e.g. a phrase with 16 words with NGRAM=5, we have:
    #     w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15
    #     overlap=False return
    #     [w0 w1 w2 w3 w4, w5 w6 w7 w8 w9, w10 w11 w12 w13 w14, w15]
    #     overlap=True return
    #     [w2 w3 w4 w5 w6, w7 w8 w9 w10 w11, w12 w13 w14 w15]
    #     with the lask segment, we choose the longest one -> w12 w13 w14 w15
    #     '''
    #     inputs, masks = self._process_custom(paragraph, NGRAM)
    #     inputs_overlap, masks_overlap, masks = self._process_overlap(NGRAM, inputs, masks)
    #     return inputs, masks, inputs_overlap, masks_overlap

    # def predict_overlap(self, paragraph, NGRAM):
    #     origin, masks, inputs_overlap, masks_overlap = self.process_overlap(paragraph, NGRAM)
    #     # preprocess and translate
    #     inputs_processed = list(np.array(origin)[masks])
    #     inputs_overlap_processed = list(np.array(inputs_overlap)[masks_overlap])
    #     if len(inputs_processed) == 0:
    #         return paragraph
    #     model_input = self.preprocess(inputs_processed + inputs_overlap_processed)
    #     model_output = self._predict(model_input)
    #     outputs = model_output[:len(inputs_processed)]
    #     outputs_overlap = model_output[len(inputs_processed):]
    #     # TODO fixed head and tail problem
    #     # results = ""
    #     results = origin[0]  # init result at 0th index
    #     # idx = 1 if masks[0] else 0
    #     # idx_overlap = 1 if masks_overlap[0] else 0
    #     idx = 1
    #     idx_overlap = 0
    #     for i in range(1, len(masks) - 1):
    #         if masks[i]:
    #             # if masks_overlap[i]:
    #             #     # left overlap compare
    #             #     if outputs[idx].split()[int(NGRAM / 2):] == outputs_overlap[idx_overlap].split()[:int(NGRAM / 2) + 1]:
    #             #         results += " " + outputs[idx]
    #             #     else:  # if not match take the origin
    #             #         space_or_not = ' ' if origin[i].strip() not in no_space_char else ''
    #             #         results += space_or_not + origin[i].strip()
    #             #     idx_overlap += 1
    #             if masks_overlap[i]:
    #                 idx_overlap += 1
    #                 # right overrlap compare
    #                 if outputs_overlap[idx_overlap - 1].split()[int(NGRAM / 2) + 1:] == outputs[idx].split()[:int(NGRAM / 2)] and \
    #                         outputs[idx].split()[int(NGRAM / 2):] == outputs_overlap[idx_overlap].split()[:int(NGRAM / 2) + 1]:
    #                     results += " " + outputs[idx]
    #                 else:  # if not match then take the origin
    #                     space_or_not = ' ' if origin[i].strip() not in no_space_char else ''
    #                     results += space_or_not + origin[i].strip()
    #                 # idx_overlap += 1
    #             else:  # if not mask_overlap then skip the compare and take output
    #                 results += " " + outputs[idx]
    #             idx += 1
    #             # idx_overlap += 1
    #         else:
    #             # results.append(origin[i].strip())  # we don't add space as it is a specialchar
    #             space_or_not = ' ' if origin[i].strip() not in no_space_char else ''
    #             results += space_or_not + origin[i].strip()
    #     # if masks[-1]:
    #     #     if len(origin[-1].split()) < NGRAM and masks_overlap[-1]:
    #     #         results += " " + outputs_overlap[-1][-len(outputs[-1]):]
    #     #     else:
    #     #         results += " " + outputs[-1]
    #     # else:
    #     #     space_or_not = ' ' if origin[-1].strip() not in no_space_char else ''
    #     #     results += space_or_not + origin[-1].strip()
    #     if len(masks) != 1:
    #         space_or_not = ' ' if origin[-1].strip() not in no_space_char else ''
    #         results += space_or_not + origin[-1].strip()
    #     return results.strip()

    def preprocess(self, text):
        '''
        convert str sentence to tensor of shape 1 x L = 1 sentence x sentence length
        or convert list of N sentences to tensor of shape N x L = N x max sentence length
        Note: self.vocab.encode by characters (e.g. đếm ngược -> [1, 46, 68, 96, 11, 34, 43, 56, 87,12, 2])
        output: tensor of shape NxL
        '''
        if isinstance(text, str):
            src_text = self.vocab.encode(text)
            src_text = np.expand_dims(src_text, axis=0)
            # return torch.LongTensor(src_text).to(self.device)  # add to(self.device)
        elif isinstance(text, list):
            src_text = []
            MAXLEN = max(len(txt) for txt in text) + 2  # 2 for start and stop token
            for txt in text:
                txt = self.vocab.encode(txt)
                text_len = len(txt)
                src = np.concatenate((txt, np.zeros(MAXLEN - text_len, dtype=np.int32)))
                src_text.append(src)
        else:
            raise TypeError('text has to be str or list of str')
        return torch.LongTensor(np.array(src_text)).to(self.device)

    def extract_phrase(self, paragraph):
        # extract phrase
        return re.findall(r'\w[\w ]*|\s\W+|\W+', paragraph)

    def process(self, paragraph, NGRAM):
        '''
        input:
            convert paragraph to phrases, skip phrares has <2 words or not match regex
            NGRAM = number of words to break too long sentences (x2 NGRAMS words) to shorten sentences (NGRAM- 2 NGRAMS words)
        output:
            intpus: list of sentences
            masks: True = valid sentence for training, False = too short or special char sentence
        '''
        phrases = self.extract_phrase(paragraph)
        inputs = []
        masks = []
        # group by n-grams
        for phrase in phrases:
            words = phrase.split()
            if len(words) < 2 or not re.match("\w[\w ]+", phrase):
                inputs.append(phrase)
                masks.append(False)
            else:
                for i in range(0, len(words), NGRAM):
                    inputs.append(' '.join(words[i:i + NGRAM]))
                    masks.append(True)
                    if len(words) - i - NGRAM < NGRAM:
                        inputs[-1] += ' ' + ' '.join(words[i + NGRAM:])
                        inputs[-1] = inputs[-1].strip()
                        break
        return inputs, masks

    def predict(self, paragraph, NGRAM=5):
        inputs, masks = self.process_custom(paragraph, NGRAM)

        # preprocess and translate
        inputs_processed = list(np.array(inputs)[masks])
        if len(inputs_processed) == 0:
            return paragraph
        model_input = self.preprocess(inputs_processed)
        model_output = self._predict(model_input)
        results = ""
        idx = 0
        # # TODO fixed head and tail problem
        # results = inputs[0]
        # idx = 1 if masks[0] else 0
        for i in range(0, len(masks) - 1):
            if masks[i]:
                results += " " + model_output[idx]
                idx += 1
            else:
                # space_or_not = ' ' if inputs[i].strip() not in no_space_char else ''
                space_or_not = ' ' if ''.join(inputs[i].split()).isalnum(
                ) or inputs[i].strip() not in no_space_char else ''
                results += space_or_not + inputs[i].strip()
        if len(masks) != 1:
            # space_or_not = ' ' if inputs[-1].strip() not in no_space_char else ''
            space_or_not = ' ' if ''.join(inputs[-1].split()).isalnum() or inputs[i] not in no_space_char else ''
            # space_or_not = ' ' if masks[-1] else ''
            results += space_or_not + inputs[-1].strip()
        return results.strip()

    # def predict_overlap(self, paragraph, NGRAM=5):
    #     # get overlap paragraph
    #     pivot = int(NGRAM / 2)
    #     cnt_word = 0
    #     iter_ = 0
    #     lwords = paragraph.split()
    #     skip_words = []
    #     while cnt_word < pivot:
    #         skip_words.append(lwords[iter_])
    #         if len(lwords[iter_]) > 1 or lwords[iter_].isalnum():
    #             cnt_word += 1
    #         iter_ += 1
    #     para_overlap = paragraph[len(' '.join(skip_words)):].strip()
    #     # predict both
    #     inputs, masks = self.process_custom(paragraph, NGRAM)
    #     inputs_overlap, masks_overlap = self.process_custom(para_overlap, NGRAM)
    #     inputs = self.preprocess(list(np.array(inputs)[masks]))
    #     inputs_overlap = self.preprocess(list(np.array(inputs_overlap)[masks_overlap]))
    #     output = self._predict(inputs)
    #     output_overlap = self._predict(inputs_overlap)
    #
    #     results = ' '.join(skip_words)
    #     idx = 0
    #     for i, mask_overlap in enumerate(masks_overlap):
    #         if mask_overlap and masks[i + len(skip_words)] and output_overlap[idx] == output[idx + len(skip_words)]:
    #             results += ' ' + output_overlap[idx]
    #             idx += 1
    #         else:
    #             space_or_not = ' ' if ''.join(inputs[-1].split()).isalnum() or inputs[i] not in no_space_char else ''
    #             results += space_or_not + inputs_overlap[i]
    #     return results.strip()
    def predict_overlap(self, paragraph, NGRAM=5):
        # save space indices for later use
        space_indices = [i for i, x in enumerate(paragraph) if x == " "]
        lwords = sum([p.split() for p in self.extract_phrase(paragraph)], [])
        # # correct randomcase
        # correct_randomcase(lwords)

        # get overlap paragraph
        pivot = int(NGRAM / 2)
        cnt_word = 0
        iter_ = 0
        skip_words = []
        try:
            while cnt_word < pivot:
                skip_words.append(lwords[iter_])
                # if len(lwords[iter_]) > 1 or lwords[iter_].isalnum():
                if lwords[iter_].isalnum():
                    cnt_word += 1
                iter_ += 1
        except IndexError:
            return paragraph  # skip too short paragraph
        if len([word for word in lwords[len(skip_words):] if word.isalnum()]) <= NGRAM:
            return paragraph  # skip too short paragraph overlap

        # pad from skip_words
        pad_lwords = 0
        while ''.join(skip_words) != ''.join(lwords[:pad_lwords]):
            pad_lwords += 1
        pad_para_overlap = 0
        while ''.join(skip_words) != paragraph[:pad_para_overlap].replace(' ', ''):
            pad_para_overlap += 1
        para_overlap = paragraph[pad_para_overlap:].strip()

        # predict
        res = self.predict(paragraph, NGRAM)
        res_overlap = self.predict(para_overlap, NGRAM)
        lres = sum([p.split() for p in self.extract_phrase(res)], [])
        lres_overlap = sum([p.split() for p in self.extract_phrase(res_overlap)], [])

        # concate overlap
        lresults = [word for word in skip_words]
        for i in range(len(lres_overlap)):
            if lres_overlap[i] == lres[i + pad_lwords]:
                lresults.append(lres_overlap[i])
            else:
                lresults.append(lwords[i + pad_lwords])

        # convert lwords to string with space
        results = ''
        space_cnt = 0
        for word in lresults:
            results += word
            if space_cnt < len(space_indices) and len(results) == space_indices[space_cnt]:
                results += ' '
                space_cnt += 1
        return results.strip()

    # def predict_overlap_(self, paragraph, NGRAM=5):
    #     # save space indices for later use
    #     space_indices = [i for i, x in enumerate(paragraph) if x == " "]
    #     if len(space_indices) <= 5:  # 6 worđs
    #         return paragraph
    #     # # correct randomcase
    #     # correct_randomcase(lwords)

    #     # predict
    #     res = self.predict(paragraph, NGRAM)
    #     lres = sum([p.split() for p in self.extract_phrase(res)], [])
    #     # convert lwords to string with space
    #     results = ''
    #     space_cnt = 0
    #     for word in lres:
    #         results += word
    #         if space_cnt < len(space_indices) and len(results) == space_indices[space_cnt]:
    #             results += ' '
    #             space_cnt += 1
    #     return results.strip()

    def _predict(self, model_input):
        model_output = translate(model_input, self.model, self.device).tolist()
        model_output = self.vocab.batch_decode(model_output)
        return model_output

    def batch_process(self, paragraphs, NGRAM):
        inputs = []
        masks = []
        para_len = []
        for p in paragraphs:
            phrases = self.extract_phrase(p)
            cnt = 0
            for phrase in phrases:
                words = phrase.split()
                if len(words) < 2 or not re.match("\w[\w ]+", phrase):
                    inputs.append(phrase.strip())
                    masks.append(False)
                    cnt += 1
                else:
                    for i in range(0, len(words), NGRAM):
                        inputs.append(' '.join(words[i:i + NGRAM]))
                        masks.append(True)
                        cnt += 1
                        if len(words) - i - NGRAM < NGRAM:
                            inputs[-1] += ' ' + ' '.join(words[i + NGRAM:])
                            inputs[-1] = inputs[-1].strip()
                            break
            para_len.append(cnt)
        return inputs, masks, para_len

    def batch_predict(self, paragraphs, NGRAM=5, batch_size=256):
        inputs, masks, para_len = self.batch_process(paragraphs, NGRAM)
        outputs = list()

        # build cluster
        print(list(np.array(inputs)[masks]))
        cluster_texts, indices = self.build_cluster_texts(list(np.array(inputs)[masks]))

        # preprocess and translate
        for _, batch_texts in cluster_texts.items():
            if len(batch_texts) <= batch_size:
                model_input = self.preprocess(batch_texts)
                model_output = self._predict(model_input)
                outputs.extend(model_output)
            else:
                for i in range(0, len(batch_texts), batch_size):
                    model_input = self.preprocess(batch_texts[i:i + batch_size])
                    model_output = self._predict(model_input)
                    outputs.extend(model_output)

        # sort result correspond to indices
        z = zip(outputs, indices)
        outputs = sorted(z, key=lambda x: x[1])
        outputs, _ = zip(*outputs)
        print('-----------------')
        print(outputs)

        # group n-grams -> final paragraphs
        para_idx = 0
        sentence_idx = 0
        paragraphs = []
        p = ""
        for i, mask in enumerate(masks):
            if para_len[para_idx] == i:
                paragraphs.append(p.strip())
                p = ""
                para_idx += 1

            if mask:
                p += " " + outputs[sentence_idx]
                sentence_idx += 1
            else:
                p += inputs[i].strip()

        return paragraphs

    @ staticmethod
    def sort_width(texts):
        batch = list(zip(texts, range(len(texts))))
        sorted_texts = sorted(batch, key=len, reverse=False)
        sorted_texts, indices = list(zip(*sorted_texts))

        return sorted_texts, indices

    def build_cluster_texts(self, texts):
        cluster_texts = defaultdict(list)
        sorted_texts, indices = self.sort_width(texts)

        for text in sorted_texts:
            cluster_texts[get_bucket(len(text))].append(text)

        return cluster_texts, indices


if __name__ == '__main__':
    model_predictor = Predictor(device='cpu', model_type='seq2seq', weight_path='weights/seq2seq_0.pth')
    unacc_paragraphs = [
        "Trên cơsở kếT quả kiểm tra hiện trạng, \
        các cơ sở nhà, đất thuộc phám vi Quản lý, gửi lấy ý kiến của Ủy ban nhân dân cấp tỉnh nơi có nhà, đất. \
        Riêng đổi với việc tổ chức kiểm tra hiện trạng, lập phương án, phê duyệt phương án sắp xếp lại, xử lý nhà, \
        đất trên địa bàn các thành phố Hà Nội, Hồ Chí Minh, Đà Nẵng, Cần Thơ, Hải Phòng do Bộ Tài chính thực hiện theo \
        quy định tại Mục 3, Điều 5 Nghị định số 167/2017/NĐ-CP: ",

        "Căn cứ Quyếtđịnh số 676/2016/QĐ-TANDTC-KHTC ngày 13/9/2016 của Chánh án Tòa án nhân dân tối cao về \
        việc phân cấp quản lý ngân sách Nhà nước và quản lý dự án đầu tư xây dựng công trình trụ sở làm việc \
        Tòa án địa phương Để đảm bảo quản lý, sử dụng có hiệu quả các cơ sở nhà, đất trong hệ thống Tòa án nhân dân, \
        Tòa án nhân dân tối cao yêu cầu Thủ trưởng các đơn vị quán triệt, nghiêm túc thực hiện Luật Quản lý, sử dụng \
        tài sản công, các văn bản pháp luật có liên quan và hướng dan trình tự, thủ tục sắp xếp lại, xử lý nhà, đat \
        như sau: ",
        "ĐIỀU 6 : ĐIỀU KHOẢN VỀ BẢO DƯỜNG, SỬA CHỮA NHÀ & CÁC TRANG THIẾT BỊ"
    ]
    # inputs, masks = model_predictor.process(unacc_paragraphs[0].strip(), NGRAM=5)
    # model_input = model_predictor.preprocess(list(np.array(inputs)[masks]))
    # model_output = model_predictor._predict(model_input)
    # print('test')
    print(model_predictor.predict(unacc_paragraphs[0].strip()))
