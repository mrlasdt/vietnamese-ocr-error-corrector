import re


STANDARDIZE_TRAIN_ADDRESS_UNITS = {  # replace value by key (not key by value)
    'đ': 'đường',
    'ql': 'quốc lộ',
    'p': 'phường',
    'f': 'phường',
    'ph': 'phường',
    'q': 'quận',
    'tp': 'thành phố',
    't.p': 'thành phố',
    "t": "tỉnh",
    'h': 'huyện',
    'p': 'phường',
    'x': 'xã',
    'tx': 'thị xã',
    't.x': 'thị xã',
    'kp': 'khu phố',
    'k.p': 'khu phố',
    'kdt': 'khu đô thị',
    'kđt': 'khu đô thị',
    'ktt': 'khu tập thể',
    'kcn': 'khu công nghiệp',
    'kcx': 'khu chế xuất',
    'kdc': 'khu dân cư',
    'tdp': 'tổ dân phố',
    'ccn': 'cụm công nghiệp',
    'kv': 'khu vực',
    'tt': 'thị trấn',
    't.t': 'thị trấn',
    'vp': 'văn phòng',
    'hn': 'hà nội',
    'hcm': 'hồ chí minh',
    'vn': 'việt nam',
}
STANDARDIZE_TRAIN_ADDRESS_SHORTKEYS = [k for k in STANDARDIZE_TRAIN_ADDRESS_UNITS if k not in ['hn', 'hcm']]

def remove_reduntant_space_and_newline(text):
    text = re.sub(r"[\r\n\t]", " ", text)  # remove newline and enter char
    text = re.sub(r"\s{2,}", " ", text)  # replace two or more sequential spaces with 1 space
    return text.strip()
    
def replace_text_by_span(text, text_replace, span):
    return text[:span[0]] + text_replace + text[span[1]:]


def add_space_before_num_in_code(text):
    for i, char in enumerate(text):
        if char.isdigit():
            # return text[:i] + "_" + text[i:]
            return text[:i] + " " + text[i:]


def standardize_shortkeys_with_number(text):
    '''
    e.g. q3 -> q 3, q.10 -> q. 10
    '''
    shortkeys = '|'.join(STANDARDIZE_TRAIN_ADDRESS_SHORTKEYS).replace('.', '\.')
    regex = "\b({s})\.?([1-9]|1[1-9])\w?\b".format(s=shortkeys)
    regex = re.compile(regex)
    matches = regex.finditer(text)
    for i, match in enumerate(matches):
        span = match.span()
        text_replace = add_space_before_num_in_code(match.group())
        # add one i to match with the additional space
        text = replace_text_by_span(text, text_replace, (span[0] + i, span[1] + i))
    # some outliers
    text = text.replace('tphcm', 'tp hcm')  # some outliers
    return text


def add_space_after_dot(text):
    for i, char in enumerate(text):
        if char == '.':
            # return text[:i] + "_" + text[i:]
            return text[:i + 1] + " " + text[i + 1:]


def standardize_shortkeys_with_text(text):
    '''
    e.g. tphcm -> tp hcm, tp.hcm -> tp. hcm
    '''
    shortkeys = '|'.join(STANDARDIZE_TRAIN_ADDRESS_SHORTKEYS).replace('.', '\.')
    regex = fr"\b({shortkeys})\.{{1}}\w+\b"
    regex = re.compile(regex)
    matches = regex.finditer(text)
    for i, match in enumerate(matches):
        span = match.span()
        text_replace = add_space_after_dot(match.group())
        # add one i to match with the additional space
        text = replace_text_by_span(text, text_replace, (span[0] + i, span[1] + i))
    # some outliers
    text = text.replace('tphcm', 'tp hcm')  # some outliers
    return text


def standardize_unit(text):
    """ 
    e.g. q 3 -> quận 3, tp hn, -> thành phố hà nội,
    """
    lts = []
    for text_split in text.split(','):
        lw = []
        for word in text_split.strip().split():
            if word in list(STANDARDIZE_TRAIN_ADDRESS_UNITS.keys()):
                lw.append(STANDARDIZE_TRAIN_ADDRESS_UNITS[word])
            elif word in [k + '.' for k in list(STANDARDIZE_TRAIN_ADDRESS_UNITS.keys())]:
                lw.append(STANDARDIZE_TRAIN_ADDRESS_UNITS[word[:-1]])
            else:
                lw.append(word)
        lts.append(lw)
    text_splited = [' '.join(ts) for ts in lts]
    text_with_comma = ", ".join(text_splited)
    return text_with_comma
