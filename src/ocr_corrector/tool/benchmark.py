import re
from rapidfuzz.distance import Levenshtein
from difflib import SequenceMatcher

def is_type_list(x, type):

    if not isinstance(x, list):
        return False

    return all(isinstance(item, type) for item in x)

class OcrMetric:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def post_processing(text):
        '''
        - Remove special characters and  extra spaces + lower case
        '''

        text = re.sub(r"[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789 ]", " ", text)
        text = re.sub(r"\s\s+", " ", text)
        text = text.strip()

        return text
    
    @staticmethod
    def cal_true_positive_char(pred, gt):
        """Calculate correct character number in prediction.
        Args:
            pred (str): Prediction text.
            gt (str): Ground truth text.
        Returns:
            true_positive_char_num (int): The true positive number.
        """

        all_opt = SequenceMatcher(None, pred, gt)
        true_positive_char_num = 0
        for opt, _, _, s2, e2 in all_opt.get_opcodes():
            if opt == 'equal':
                true_positive_char_num += (e2 - s2)
            else:
                pass
        return true_positive_char_num

    @staticmethod 
    def count_matches(pred_texts, gt_texts, use_ignore=False):
        """Count the various match number for metric calculation.
        Args:
            pred_texts (list[str]): Predicted text string.
            gt_texts (list[str]): Ground truth text string.
        Returns:
            match_res: (dict[str: int]): Match number used for
                metric calculation.
        """
        match_res = {
            'gt_char_num': 0,
            'pred_char_num': 0,
            'true_positive_char_num': 0,
            'gt_word_num': 0,
            'match_word_num': 0,
            'match_word_ignore_case': 0,
            'match_word_ignore_case_symbol': 0,
            'match_kie': 0,
            'match_kie_ignore_case': 0

        }
        # comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
        # comp = re.compile('[]')
        norm_ed_sum = 0.0

        gt_texts_for_ned_word = []
        pred_texts_for_ned_word = []
        for pred_text, gt_text in zip(pred_texts, gt_texts):
            if gt_text == pred_text:
                match_res['match_word_num'] += 1
                match_res['match_kie'] += 1
            gt_text_lower = gt_text.lower()
            pred_text_lower = pred_text.lower()
            if gt_text_lower == pred_text_lower:
                match_res['match_word_ignore_case'] += 1

            # gt_text_lower_ignore = comp.sub('', gt_text_lower)
            # pred_text_lower_ignore = comp.sub('', pred_text_lower)
            if use_ignore:
                gt_text_lower_ignore = OcrMetric.post_processing(gt_text_lower)
                pred_text_lower_ignore = OcrMetric.post_processing(pred_text_lower)

            else:
                gt_text_lower_ignore = gt_text_lower
                pred_text_lower_ignore = pred_text_lower

            if gt_text_lower_ignore == pred_text_lower_ignore:
                match_res['match_kie_ignore_case'] += 1

            gt_texts_for_ned_word.append(gt_text_lower_ignore.split(" "))
            pred_texts_for_ned_word.append(pred_text_lower_ignore.split(" "))

            match_res['gt_word_num'] += 1

            norm_ed = Levenshtein.normalized_distance(pred_text_lower_ignore,
                                                    gt_text_lower_ignore)
            # if norm_ed > 0.1:
            #     print(gt_text_lower_ignore, pred_text_lower_ignore, sep='\n')
            #     print("-"*20)
            norm_ed_sum += norm_ed

            # number to calculate char level recall & precision
            match_res['gt_char_num'] += len(gt_text_lower_ignore)
            match_res['pred_char_num'] += len(pred_text_lower_ignore)
            true_positive_char_num = OcrMetric.cal_true_positive_char(
                pred_text_lower_ignore, gt_text_lower_ignore)
            match_res['true_positive_char_num'] += true_positive_char_num

        normalized_edit_distance = norm_ed_sum / max(1, len(gt_texts))
        match_res['ned'] = normalized_edit_distance # type: ignore

        # NED for word-level
        norm_ed_word_sum = 0.0
        # print(pred_texts_for_ned_word[0])
        unique_words = list(
            set([x for line in pred_texts_for_ned_word for x in line] + [x for line in gt_texts_for_ned_word for x in line]))
        preds = [[unique_words.index(w) for w in pred_text_for_ned_word]
                for pred_text_for_ned_word in pred_texts_for_ned_word]
        truths = [[unique_words.index(w) for w in gt_text_for_ned_word] for gt_text_for_ned_word in gt_texts_for_ned_word]
        for pred_text, gt_text in zip(preds, truths):
            norm_ed_word = Levenshtein.normalized_distance(pred_text,
                                                        gt_text)
            if norm_ed_word < 0.2:
                print(pred_text, gt_text)
            norm_ed_word_sum += norm_ed_word

        normalized_edit_distance_word = norm_ed_word_sum / max(1, len(gt_texts))
        match_res['ned_word'] = normalized_edit_distance_word  # type: ignore

        return match_res

    @staticmethod
    def eval_ocr_metric(pred_texts, gt_texts, metric='acc'):
        """Evaluate the text recognition performance with metric: word accuracy and
        1-N.E.D. See https://rrc.cvc.uab.es/?ch=14&com=tasks for details.
        Args:
            pred_texts (list[str]): Text strings of prediction.
            gt_texts (list[str]): Text strings of ground truth.
            metric (str | list[str]): Metric(s) to be evaluated. Options are:
                - 'word_acc': Accuracy at word level.
                - 'word_acc_ignore_case': Accuracy at word level, ignoring letter
                case.
                - 'word_acc_ignore_case_symbol': Accuracy at word level, ignoring
                letter case and symbol. (Default metric for academic evaluation)
                - 'char_recall': Recall at character level, ignoring
                letter case and symbol.
                - 'char_precision': Precision at character level, ignoring
                letter case and symbol.
                - 'one_minus_ned': 1 - normalized_edit_distance
                In particular, if ``metric == 'acc'``, results on all metrics above
                will be reported.
        Returns:
            dict{str: float}: Result dict for text recognition, keys could be some
            of the following: ['word_acc', 'word_acc_ignore_case',
            'word_acc_ignore_case_symbol', 'char_recall', 'char_precision',
            '1-N.E.D'].
        """
        assert isinstance(pred_texts, list)
        assert isinstance(gt_texts, list)
        assert len(pred_texts) == len(gt_texts)

        assert isinstance(metric, str) or is_type_list(metric, str)
        if metric == 'acc' or metric == ['acc']:
            metric = [
                'word_acc', 'word_acc_ignore_case', 'word_acc_ignore_case_symbol',
                'char_recall', 'char_precision', 'one_minus_ned', "one_minus_ned_word"
            ]
        metric = set([metric]) if isinstance(metric, str) else set(metric)

        # supported_metrics = set([
        #     'word_acc', 'word_acc_ignore_case', 'word_acc_ignore_case_symbol',
        #     'char_recall', 'char_precision', 'one_minus_ned', 'one_minust_ned_word'
        # ])
        # assert metric.issubset(supported_metrics)

        match_res = OcrMetric.count_matches(pred_texts, gt_texts)
        eps = 1e-8
        eval_res = {}

        if 'char_recall' in metric:
            char_recall = 1.0 * match_res['true_positive_char_num'] / (
                eps + match_res['gt_char_num'])
            eval_res['char_recall'] = char_recall

        if 'char_precision' in metric:
            char_precision = 1.0 * match_res['true_positive_char_num'] / (
                eps + match_res['pred_char_num'])
            eval_res['char_precision'] = char_precision

        if 'word_acc' in metric:
            word_acc = 1.0 * match_res['match_word_num'] / (
                eps + match_res['gt_word_num'])
            eval_res['word_acc'] = word_acc

        if 'word_acc_ignore_case' in metric:
            word_acc_ignore_case = 1.0 * match_res['match_word_ignore_case'] / (
                eps + match_res['gt_word_num'])
            eval_res['word_acc_ignore_case'] = word_acc_ignore_case

        if 'word_acc_ignore_case_symbol' in metric:
            word_acc_ignore_case_symbol = 1.0 * match_res[
                'match_word_ignore_case_symbol'] / (
                    eps + match_res['gt_word_num'])
            eval_res['word_acc_ignore_case_symbol'] = word_acc_ignore_case_symbol

        if 'one_minus_ned' in metric:

            eval_res['1-N.E.D'] = 1.0 - match_res['ned']

        if 'one_minus_ned_word' in metric:

            eval_res['1-N.E.D_word'] = 1.0 - match_res['ned_word']

        if 'line_acc_ignore_case_symbol' in metric:
            line_acc_ignore_case_symbol = 1.0 * match_res[
                'match_kie_ignore_case'] / (
                    eps + match_res['gt_word_num'])
            eval_res['line_acc_ignore_case_symbol'] = line_acc_ignore_case_symbol

        if 'line_acc' in metric:
            word_acc_ignore_case_symbol = 1.0 * match_res[
                'match_kie'] / (
                    eps + match_res['gt_word_num'])
            eval_res['line_acc'] = word_acc_ignore_case_symbol

        for key, value in eval_res.items():
            eval_res[key] = float('{:.4f}'.format(value))

        return eval_res

if __name__ == "__main__":
    res = OcrMetric.eval_ocr_metric(["Tooi ddi hocj", "Sam hoi"], ["Tôi đi học", "Sam hoi"])
    print(res)
