import re
import numpy as np
import unidecode
from ocr_corrector.tool.common import chars_regrex, same_chars, most_wrong_accent_chars, most_wrong_chars_regrex, most_wrong_distribution, SEED, most_wrong_chance, common_randomcase

np.random.seed(SEED)


def _char_regrex(text, most_wrong_mode):
    '''
    find matched chars in chars_regrex (e.g. 'cơ' -> 'ơ')
    '''
    if most_wrong_mode:
        match_chars = re.findall(most_wrong_chars_regrex, text)
    else:
        match_chars = re.findall(chars_regrex, text)
    return match_chars


def change(text, match_chars, most_wrong_mode):
    if most_wrong_mode:
        char_dist = [most_wrong_distribution[char] for char in match_chars]
        norm = [float(i) / sum(char_dist) for i in char_dist]
        replace_char = np.random.choice(match_chars, 1, p=norm)[0]
        insert_chars = most_wrong_accent_chars[replace_char]
    else:
        replace_char = match_chars[np.random.randint(low=0, high=len(match_chars), size=1)[0]]
        insert_chars = same_chars[unidecode.unidecode(replace_char)]
    insert_char = insert_chars[np.random.randint(low=0, high=len(insert_chars), size=1)[0]]
    text = text.replace(replace_char, insert_char, 1)
    return text


def get_matched_chars_and_mask(lwords, most_wrong_mode):
    llchars_matched = list(map(lambda w: _char_regrex(w, most_wrong_mode), lwords))
    lmask_matched = list(map(bool, llchars_matched))
    lmask_matched_nonzero = np.nonzero(lmask_matched)[0]
    return llchars_matched, lmask_matched_nonzero


def random_replace_accent(text, ratio=0.2):
    lwords = text.split()
    if len(lwords) < 2:  # only permutate sentence with 2+ words
        return text
    most_wrong_mode = True if np.random.random() < most_wrong_chance else False
    llchars_matched, lmask_matched_nonzero = get_matched_chars_and_mask(lwords, most_wrong_mode)
    if len(lmask_matched_nonzero) == 0 and most_wrong_mode:
        most_wrong_mode = False
        llchars_matched, lmask_matched_nonzero = get_matched_chars_and_mask(lwords, most_wrong_mode)
    if len(lmask_matched_nonzero) == 0:
        return text
    lmask_ratio = np.random.choice(lmask_matched_nonzero, size=max(1, int(ratio * len(llchars_matched))))
    for i in lmask_ratio:
        lwords[i] = change(lwords[i], llchars_matched[i], most_wrong_mode)
    return ' '.join(lwords)


if __name__ == "__main__":
    text = "Trên cơ sở kết quả kiểm tra hiện trạng"
    # text = 'ngày mai tôi đi học'
    text = 'trễ và sự cố'
    print(random_replace_accent(text))

