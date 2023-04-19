from src.tool.utils import extract_phrases
import itertools
from tqdm import tqdm
from ocr_corrector.tool.common import NGRAMS, SEED
import re
import lmdb
from unicodedata import normalize
import random
random.seed(SEED)

# load data from file txt
data_folder = '.data'
corpus_file = f'{data_folder}/corpus-full.txt'
wiki_file = f'{data_folder}/train_tieng_viet.txt'

n_sentences = 6000000
with open(corpus_file, 'r') as f:
    lines = f.readlines()[:n_sentences]  # select only first 6M sentences
with open(wiki_file, 'r') as f:
    lines.extend(f.readlines()[:n_sentences])
random.shuffle(lines)


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


cache = {}
error = 0
val_size = int(0.2 * len(lines))

# open lmdb database
train_env = lmdb.open(f'{data_folder}/train_lmdb', map_size=10099511627776)
val_env = lmdb.open(f'{data_folder}/val_lmdb', map_size=10099511627776)

phrases = itertools.chain.from_iterable(extract_phrases(text) for text in lines)
char_regrex = '^[_aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789 !\"\',\-\.:;?_\(\)]+$'
train_cnt = 0
val_cnt = 0
tgt_env = val_env

print('Creating dataset...')
for p in tqdm(phrases):  # , desc=f'Creating dataset...'):
    if not re.match(char_regrex, p):  # make sure vietnamese only
        continue
    words = p.strip().split()
    if len(words) < 2:  # skip single word sentence
        continue
    for i in range(0, len(words), NGRAMS):
        if len(words) - i < NGRAMS:
            if len(words) - i < 2:  # skip single words leftover
                continue
            ngram_text = ' '.join(words[i:])  # else
        else:
            ngram_text = ' '.join(words[i:i + NGRAMS])
        if val_cnt == val_size:  # change val to train env
            if len(cache) > 0:
                writeCache(tgt_env, cache)
                cache = {}
            tgt_env = train_env
        if val_cnt < val_size:
            # write data
            textKey = 'text-%12d' % val_cnt
            val_cnt += 1
        else:
            # write data
            textKey = 'text-%12d' % train_cnt
            train_cnt += 1

        ngram_text = ngram_text.strip()
        ngram_text = ngram_text.rstrip()
        ngram_text = normalize("NFC", ngram_text)

        cache[textKey] = ngram_text.encode()
        if len(cache) % 1000 == 0:
            writeCache(tgt_env, cache)
            cache = {}

if len(cache) > 0:
    writeCache(tgt_env, cache)

cache = {}
cache['num-samples'] = str(train_cnt).encode()
writeCache(train_env, cache)

cache = {}
cache['num-samples'] = str(val_cnt).encode()
writeCache(val_env, cache)
