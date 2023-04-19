from src.tool.utils import extract_phrases
import itertools
from tqdm import tqdm
from ocr_corrector.tool.common import NGRAMS, SEED
import re
import lmdb
from unicodedata import normalize
import random
random.seed(SEED)
from multiprocessing import Pool, Value, Manager
NCPU = 16


class Counter(object):
    # https://stackoverflow.com/questions/2080660/how-to-increment-a-shared-counter-from-multiple-processes
    def __init__(self):
        self.val = Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


data_folder = 'dataloader/data'
corpus_file = f'{data_folder}/corpus-full.txt'
wiki_file = f'{data_folder}/train_tieng_viet.txt'

n_sentences = 12 * 1e6
with open(corpus_file, 'r') as f:
    lines = f.readlines()[:int(n_sentences / 2)]  # select only first 6M sentences
with open(wiki_file, 'r') as f:
    lines.extend(f.readlines()[:int(n_sentences / 2)])
random.shuffle(lines)

# cache = {}
cache = Manager().dict()
# cache = Value('cache')
val_size = int(0.2 * len(lines))

# open lmdb database

train_env = lmdb.open(f'{data_folder}/train_lmdb_par', map_size=10099511627776)
val_env = lmdb.open(f'{data_folder}/val_lmdb_par', map_size=10099511627776)

phrases = itertools.chain.from_iterable(extract_phrases(text) for text in lines)
char_regrex = '^[_aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789 !\"\',\-\.:;?_\(\)]+$'
train_cnt = Counter()
val_cnt = Counter()
tgt_env = val_env

print('Creating dataset...')


def process_phrase(p):
    global train_env, val_env, cache
    if not re.match(char_regrex, p):  # make sure vietnamese only
        return None
    words = p.strip().split()
    if len(words) < 2:  # skip single word sentence
        return None
    for i in range(0, len(words), NGRAMS):
        if len(words) - i < NGRAMS:
            if len(words) - i < 2:  # skip single words leftover
                continue
            ngram_text = ' '.join(words[i:])  # else
        else:
            ngram_text = ' '.join(words[i:i + NGRAMS])
        if val_cnt.value == val_size:  # change val to train env
            tgt_env = train_env
            if len(cache) > 0:
                writeCache(tgt_env, cache)
                cache = {}
        if val_cnt.value < val_size:
            # write data
            textKey = 'text-%12d' % val_cnt.value
            tgt_env = val_env
            val_cnt.increment()
        else:
            # write data
            textKey = 'text-%12d' % train_cnt.value
            tgt_env = train_env
            train_cnt.increment()

        ngram_text = ngram_text.strip()
        ngram_text = ngram_text.rstrip()
        ngram_text = normalize("NFC", ngram_text)

        cache[textKey] = ngram_text.encode()
        if len(cache) % 1000 == 0:
            writeCache(tgt_env, cache)
            cache = {}
        print('val_cnt', val_cnt.value, 'train_cnt', train_cnt.value)


with Pool(NCPU) as p:
    p.map(process_phrase, phrases)
    # for _ in tqdm(p.imap_unordered(process_phrase, phrases), total=len(list(phrases))):
    # pass

if len(cache) > 0:
    writeCache(tgt_env, cache)

cache = {}
cache['num-samples'] = str(train_cnt.value).encode()
writeCache(train_env, cache)

cache = {}
cache['num-samples'] = str(val_cnt.value).encode()
writeCache(val_env, cache)
