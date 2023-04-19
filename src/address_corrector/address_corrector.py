# from pathlib import Path  # add Fiintrade path to import config, required to run main()
# import sys
# FILE = Path(__file__).absolute()
# sys.path.append(FILE.parents[2].as_posix())  # add Fiintrade/ to path
# %%
import json
import pickle
# %%
import re
from sklearn.metrics import classification_report
# from srcc.tools.utils import read_json
# from underthesea import word_tokenize #vietnames tokenizer token words by semantic, such as nguyen dinh chieu -> nguyen dinh, chieu => undesirable -> use nltk instead
from nltk.tokenize import word_tokenize

from .data_preprocess_address import standardize_shortkeys_with_number, standardize_shortkeys_with_text, standardize_unit, remove_reduntant_space_and_newline
from .trie import Trie, TrieNode
import unicodedata
from unidecode import unidecode as remove_accent
import pandas as pd
import Levenshtein

VN_CHARS = unicodedata.normalize('NFC', 'àáảãạâấầẩẫậăắằẩẫặòóỏõọôồốổỗộơờớởỡợèéẻẽẹêềếểễệuùúủũụìíỉĩịýỳỷỹỵđưừứửữự')


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
# %%


class AddressCorrector:
    """
    TODO:
    - implement flexible prefix: thi trấn -> thị trấn

    """
    inf_ed = 100  # big number that edit distance cannot reach

    def __init__(self, model_path) -> None:
        self._model_path: str = model_path
        self.model_trie: Trie = self.load_model(model_path)
        self._max_address_len = 4  # TODO: valid assusmption that all address is at maximum 4 words, for example ba ria vung tau
        self.levels_dict = {
            "province": ["tỉnh", "thành phố"],
            "district": ["thành phố", "quận", "huyện", "thị xã"],
            "ward": ["xã", "phường", "thị trấn"],
        }
        # smaller is better but slower -> increase recall, precision is 1.0 (the algorithm already is very conservative)
        self.prefix_threshold = 0.6

    @staticmethod
    def load_model(model_path: str) -> Trie:
        with open(model_path, 'rb') as f:
            vnaddress_trie_dict = pickle.load(f)
        return vnaddress_trie_dict

    @staticmethod
    def preprocess(text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        text = text.lower().strip().replace("✪", " ")
        text = standardize_shortkeys_with_number(text)
        text = standardize_shortkeys_with_text(text)
        text = standardize_unit(text)
        text = remove_reduntant_space_and_newline(text)
        return text

    @staticmethod
    def join_lwords(lwords: list[str], delimiter=" "):
        return delimiter.join(lwords).strip()

    # @staticmethod
    # def extract_phrase(text) -> list[str]:
    #     return re.findall(r'[^.,]*', text)

    @staticmethod
    def rank_and_select(word, llookup: list[tuple[str, int]]) -> str:
        llookup = [l[0] for l in llookup if l[1] == llookup[0]
                   [1]]  # get all smallest edis distance look up
        word = remove_accent(word)
        llookup = sorted([(Levenshtein.distance(word, remove_accent(l)), l) for l in llookup])
        return llookup[0][::-1]

    @staticmethod
    def trie_look_up(trie, word, thresh):
        res = trie.search(word, 1 - thresh)  # word, edit distance
        return AddressCorrector.rank_and_select(word, res) if res else ("", AddressCorrector.inf_ed)
        # return res[0][0] if res and 1-res[0][1]/len(word) >= threshold else "" #res[0] = greedy, choose the minimum edit distance

    def get_lwords_before_or_after_prefix(self, s: str, prefix: str, before: bool) -> list[str]:
        if self._max_address_len > 4:
            raise NotImplementedError("Currently only support max_address_len <= 4 ")
        if not before:
            # exclusive for digit address, e.g., q1 q2 p8...
            result = re.findall(r"{0}[^a-zA-Z{1}]*(\d+)".format(prefix, VN_CHARS), s)
            if not result:
                result = re.findall(
                    r"{0}[^a-zA-Z0-9{1}]*(\w+)\s+(\w+)[\s.-]*(\w+)?[\s.-]*(\w+)?\b".format(prefix, VN_CHARS), s)
            # result = re.findall(r"{}[\s.]*(\w+)\s+(\w+)\s?(\w+)?\s?(\w+)?\b".format(prefix), s)
        else:
            result = re.findall(
                r"\b(\w+)\s+(\w+)[\s.-]*(\w+)?[\s.-]*(\w+)?[^a-zA-Z{1}]*{0}".format(prefix, VN_CHARS), s)
            # result = re.findall(r"\b(\w+)\s+(\w+)\s?(\w+)?\s?(\w+)?[\s.]*{}".format(prefix), s)
        result = [i for t in result for i in t]  # flatten the list
        return result

    def trie_lookup_address_by_prefix(self, trie, address, prefix, thresh, is_street=False):
        lookup_prefix = prefix.lower()
        lwords = self.get_lwords_before_or_after_prefix(
            address, lookup_prefix, before=is_street)  # street is placed before prefix
        lookup_address = ""
        min_ed = self.inf_ed
        if lwords:
            if lwords[0].isdigit() and not is_street:  # allow digit in street addresses
                return lookup_prefix, lwords[0]  # exclusive for digit address, e.g., q1 q2 p8...
            for len_words in range(self._max_address_len, 1, -1):  # 4 3 2
                s = self.join_lwords(lwords[:len_words]).strip() if not is_street else self.join_lwords(
                    lwords[-len_words:]).strip()  # reverse order for street address
                curr_lookup_address, curr_ed = self.trie_look_up(trie, s, thresh)
                if curr_ed < min_ed:
                    lookup_address = curr_lookup_address
                    min_ed = curr_ed
                    if is_street:
                        lookup_prefix = address.split(s)[0].strip()
        # if is_street and not lookup_address:
        #     return self.join_lwords(lwords)
        return lookup_prefix, lookup_address

    def parse_by_prefix(self, lprefix: list[str],
                        curr_dict: dict, address: str, thresh: float, is_street: bool) -> tuple[str, str]:
        """_summary_
        This function parse address after a prefix using a dict (trie) and Edit distance
        For example: Thành phố Ha noi -_ Thành phố Hà Nội
        Args:
            lprefix (list[str]): list of prefixes at specific level (province/district,ward)
            curr_dict (dict): currrent dictionary at specific level
            address (str): the address need to parse
            thresh (float): only accept parsed text with edit distance above threshold

        Returns:
            tuple[str,str]: return the parsed address and its prefĩx
        """
        lookup_address = ""
        lookup_prefix = ""
        curr_trie = curr_dict["trie_street"] if is_street else (
            curr_dict["trie"] if "trie" in curr_dict else curr_dict["trie_ward"])  # get current level trie
        for prefix in lprefix:  # loop through all prefix at every levels, for example tỉnh, thành phố
            if not lookup_address:

                curr_lookup_prefix, curr_lookup_address = self.trie_lookup_address_by_prefix(
                    curr_trie, address, prefix, thresh, is_street)
                if curr_lookup_address and not curr_lookup_address.isdigit() and not is_street:
                    # TODO: this line of code create an limitation that if the original prefix is wrong, e.g., quận hà nội -> cannot correct prefix -> maybe fix in auto correct function, not correct by prefix
                    # make sure the lookup_address have the right prefix, forexample thanh pho hai duong and tinh hai duong is not the same
                    curr_lookup_address = "" if prefix not in curr_dict[curr_lookup_address]["pre"] else curr_lookup_address
                lookup_address += curr_lookup_address
                lookup_prefix = curr_lookup_prefix
        return lookup_prefix, lookup_address

    @staticmethod
    def extract_substring_between_two_words(text: str, start: str, end: str) -> str:
        try:
            start_index = text.rfind(start)
            end_index = text.rfind(end)
            if start_index != -1 and end_index != -1 and end_index > start_index:
                return text[start_index:end_index + len(end)]
        except KeyError:
            return ""

    def parse_auto(self, curr_dict: dict, address: str, thresh: float, is_street: bool) -> tuple[str, str, str]:
        """_summary_
        This function parse address automatically using a dict (trie) and Edit distance
        For example: Quan Ba Dinh Ha noi -> Quận Ba Đình, Thành Phố Hà Nội
        Args:
            curr_dict (dict): currrent dictionary at specific level
            address (str): the address need to parse
            thresh (float): only accept parsed text with edit distance above threshold
        Returns:
            tuple[str,str]: return the parsed address and its prefĩx
        """
        # lwords = [w for w in word_tokenize(address) if w.isalpha()] #only accept alphabetical words (including vietnamese)
        # only accept alphabetical words (including vietnamese)
        lwords = [w for w in word_tokenize(address) if w.isalnum()]
        lookup_address = ""
        lookup_prefix = ""
        lookup_str = ""
        min_ed = self.inf_ed  # current min edit distance
        for len_words in range(self._max_address_len, 1, -1):  # 4 3 2
            try:
                curr_trie = curr_dict["trie_street"] if is_street else (
                    curr_dict["trie"] if "trie" in curr_dict else curr_dict["trie_ward"])  # get current level trie
                lwords_ = lwords[-len_words:]
                s = self.join_lwords(lwords_).strip()  # get last nwords
                curr_lookup_address, curr_ed = self.trie_look_up(curr_trie, s, thresh)
                if curr_ed < min_ed:
                    lookup_address = curr_lookup_address
                    min_ed = curr_ed
                    lookup_prefix = curr_dict[lookup_address]["pre"] if not is_street else address.rsplit(
                        s, 1)[0].strip()  # keep numeric street address in the prefix, same as parse_by_prefix
                    lookup_str = self.extract_substring_between_two_words(address, lwords_[0], lwords_[-1]) or s
            except IndexError:
                continue  # in case there are not enough words for the current len_words
        return lookup_prefix, lookup_address, lookup_str

    # def parse_street_by_prefix(self, address:str, lparsed_address:list[tuple[str,str]], thresh:float) -> str:
    #     province = lparsed_address[0][1]
    #     district = lparsed_address[1][1] if not lparsed_address[1][1].isdigit() else self.join_lwords(lparsed_address[1]) #exclusive for numeric address
    #     trie_street = self.model_trie[province][district]["trie_street"]
    #     street =""
    #     for prefix in self.levels_dict["ward"]:
    #         if not street:
    #             street += self.trie_lookup_address_by_prefix(trie_street, address, prefix, thresh, is_street=True)
    #     return street

    def drop_remaining_wrong_prefix(self, remain_address_after_parsed):
        """
        Example: = remain_address_after_parsed = 'xóm 1 xã quảng that, n.'
        => quang that, n. is recognized as quang nham, instead of quang thai
        """
        remain_address_after_parsed_ = remain_address_after_parsed.split(",")
        if len(remain_address_after_parsed_) > 2:
            remaining_wrong_prefix = remain_address_after_parsed_[-1].strip()
            if len(remaining_wrong_prefix) < 2:  # drop wrong prefix
                return self.join_lwords(remain_address_after_parsed_[:-1], delimiter=",")
        return remain_address_after_parsed

    def extract_remain_address_after_parsed(
            self, remain_address_after_parsed: str, split_key: str, is_parsed_auto: bool) -> str:
        remain_address_after_parsed_ = remain_address_after_parsed.rsplit(
            split_key, 1)[0]  # rsplit to reverse the order of split
        if is_parsed_auto:
            remain_address_after_parsed_ = self.drop_remaining_wrong_prefix(remain_address_after_parsed_)
        return remain_address_after_parsed_.strip()

    def _parse_address_single_level(self, lprefix, curr_dict, address, remain_address_after_parsed, thresh, is_street):
        lookup_prefix, lookup_address = self.parse_by_prefix(lprefix, curr_dict, address, thresh, is_street=is_street)
        if not lookup_address:  # try parse auto if cannot be parsed by prefix
            lookup_prefix, lookup_address, lookup_str = self.parse_auto(
                curr_dict, remain_address_after_parsed, thresh, is_street=is_street)
            if lookup_address:  # if parsed successfully by auto, update remain address
                remain_address_after_parsed = self.extract_remain_address_after_parsed(
                    remain_address_after_parsed, lookup_str, is_parsed_auto=True)
        elif not is_street:  # if parsed successfully by prefix and not is_street, update remain address by lookup_prefix
            remain_address_after_parsed = self.extract_remain_address_after_parsed(
                remain_address_after_parsed, lookup_prefix, is_parsed_auto=False)
        if is_street and not lookup_address and not lookup_prefix:  # return original street andress if cannot parse
            lookup_address = remain_address_after_parsed
        return lookup_prefix, lookup_address, remain_address_after_parsed

    @ staticmethod
    def update_trie_dict(curr_dict, lookup_address, lookup_prefix) -> dict:
        # return empty dict if cannot find address in the dict, the loop will end in the next iteration anyway
        return curr_dict.get(
            lookup_address, {}) or curr_dict.get(
            AddressCorrector.join_lwords([lookup_prefix, lookup_address]),
            {})

    def parse_address(self, address: str, thresh: float, parse_street: bool) -> list[tuple[str, str]]:
        """_summary_
        This function first try to parse the address by defined prefix, and then try parse auto if needed
        Note that street is seperately treat because 1 street can be in many district
        Args:
            address (str): The address need to be parsed
            thresh (float): only accept parsed text with edit distance above threshold
            parse_street (bool): whether parse the street info or not. When eval we don't want to take into street info since the current dictionary has little informantion about it
        Returns:
            list[tuple[str,str]]: return the parsed address and its prefix
        Algorithm Explain:
        address = '16.5 C/C 4 Nguyễn Đình Chiểu Đa Kao, Quận 1, Thành Phố Hồ Chí Minh'
        1. Use prefix "thành phố" to parse "Hồ Chí Minh" -> found "Hồ Chí Minh" in the model_trie
        2. Use prefix "quận" to parse "1". Exclusively, with numeric address, the original address is kept, i.g, no trie look up is performed
        3. No prefix found for "Đa Kao" -> Use parse_auto -> success fully parsed
        4. If parse_street = True, the street is parsed with the same algorithm:
            + First try to parse by prefix, however we will use the ward prefix and try to parse the street after it
            + If cannot be parsed by prefix, then try parse_auto
            + The remain address (16.5 C/C 4), which is usually the number of street, is keep as original and saved as a prefix
        """
        # address = self.join_lwords(word_tokenize(address)) #necessary?
        lparsed_address = []
        curr_dict = self.model_trie
        remain_address_after_parsed = address
        for i, (level, lprefix) in enumerate(self.levels_dict.items()):
            if i != len(lparsed_address) or not curr_dict:  # the address should be parsed at every level
                return []  # if cannot parsed at previous tree, return empty
            lookup_prefix, lookup_address, remain_address_after_parsed = self._parse_address_single_level(
                lprefix, curr_dict, address, remain_address_after_parsed, thresh, is_street=False)
            if level == "ward" and parse_street:  # if parse_street
                numeric_street_address, street, _ = self._parse_address_single_level(
                    lprefix, curr_dict, address, remain_address_after_parsed, thresh, is_street=True)
            if lookup_address:  # if parsed, append result to the list and update dict
                lparsed_address.append((lookup_prefix, lookup_address))
                # curr_dict = curr_dict[lookup_address] if not lookup_address.isdigit() else curr_dict[self.join_lwords([lookup_prefix, lookup_address])] #proceed dict to the next level
                curr_dict = self.update_trie_dict(curr_dict, lookup_address, lookup_prefix)
        if parse_street:
            lparsed_address.append((numeric_street_address, street))
        return lparsed_address

    def convert_lparsed_address_to_address(self, lparsed_address: list[tuple[str, str]]) -> str:
        lparsed_address = [p for p in lparsed_address if p != ("", "")]
        return self.join_lwords([self.join_lwords(p) for p in lparsed_address[::-1]], delimiter=", ")

    @staticmethod
    def handle_VN_at_the_end(s: str) -> tuple[str, str]:
        """
        s = "Số 6 Biệt Thự 2, Bán Đảo Linh Đàm, Phường Hoàng Liệt, Quận Hoàng Mai, Hà Nội, Việt Nam lô CN".lower()
        số 6 biệt thự 2, bán đảo linh đàm, phường hoàng liệt, quận hoàng mai, hà nội
        , việt nam
        """
        match = re.search(r"[,\s]*(việt)+\s*(nam)*\w*", s)
        if match:
            substring = s.split(match.group())[0]
            return substring, match.group()
        else:
            return s, ""

    def post_process_corrected_address(self, address_):
        return address_.replace(",,", ",").title()

    def correct(self, address: str, parse_street=True) -> str:
        """
        parse_street=False for evalation
        """
        address_: str = self.preprocess(address)
        address_, vn_at_the_end = self.handle_VN_at_the_end(address_)
        lparsed_address = self.parse_address(address_, thresh=self.prefix_threshold, parse_street=parse_street)
        if not lparsed_address:
            return address
        address_ = self.convert_lparsed_address_to_address(lparsed_address) + vn_at_the_end
        return self.post_process_corrected_address(address_)

    def eval(self, data_path, type_column, dexcludes={}, save_csv=False, keep_origin=False):
        """_summary_

        Args:
            data_path (_type_): _description_
            type_column (_type_): _description_
            parse_street (bool, optional): _description_. Defaults to False.
            lexcludes (list, optional): _description_. Defaults to [].
            save_csv (bool, optional): _description_. Defaults to False.
            keep_origin (bool, optional): to measure the improvement of correction
        """
        data = read_json(data_path)

        if type_column == "Invoice":
            def flatten_dict(d, parent_key='', sep='.'):
                items = []
                for k, v in d.items():
                    new_key = parent_key + sep + k if parent_key else k
                    if isinstance(v, dict) and isinstance(list(v.values())[0], dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)
            data = flatten_dict(data)
        y_true, y_pred = [], []
        ddata = {}
        for k, d in data.items():
            ori_pred = d["pred"] if not type_column == "Invoice" else d["pred"].rsplit("\t")[0]
            ori_label = d["label"] if not type_column == "Invoice" else d["label"].rsplit("\t")[0]
            if ori_pred == ori_label:  # skipp all pred that already correct
                continue
            if k == "150094748_1728953773944481_6269983404281027305_n.json":
                print("Debugging")
            pred = corrector.correct(ori_pred, parse_street=True) if not keep_origin else ori_pred
            label = corrector.correct(ori_label, parse_street=True) if not keep_origin else ori_label
            ddata[k] = {}
            data[k]["Type"] = type_column
            ddata[k]["Predict"] = ori_pred
            ddata[k]["Label"] = ori_label
            ddata[k]["Post-processed"] = pred
            ddata[k]["Class"] = pred == label if k not in dexcludes else -1
            ddata[k]["Wrong_street"] = pred != label and corrector.correct(
                ori_pred,
                parse_street=False) == corrector.correct(
                ori_label,
                parse_street=False)
            ddata[k]["Note"] = dexcludes.get(k, "")
            if ddata[k]["Class"] != -1:  # not excluded
                y_pred.append(pred == label)
                y_true.append(1)
                if pred != label:
                    print("\n", k, '-' * 50)
                    print("corrected pred ------ \t", pred)
                    print("origin pred ------ \t", ori_pred)
                    print("corrected label ------\t", label)
                    print("origin label ------\t", ori_label)
        print(classification_report(y_true, y_pred))
        if save_csv:
            df = pd.DataFrame.from_dict(ddata, orient="index")
            df.to_excel(f"address_post_processed_{type_column}.xlsx")

# def get_date_by_pattern(date_string):
#     match = re.findall(r"([^\d\s]+)?\s*(\d{1}\s*\d?\s+|\d{2}\s+|\d+\s*\b)", date_string)
#     if not match:
#         return ""
#     if len(match) > 3:
#         day = match[0][-1].replace(" ", "")
#         year = match[-1][-1].replace(" ", "")
#         # since in the VIETNAMESE DRIVER LICENSE, the tháng/month is behind the stamp and can be recognized as any thing => mistạken number may be in range (1->-3) => choose month to be -2
#         month = match[-2][-1].replace(" ", "")
#         return "/".join([day, month, year])
#     else:
#         return "/".join([m[-1].replace(" ", "") for m in match])


# %%
if __name__ == "__main__":
    # %%
    test = '16.5 C/C 4 Nguyễn Đình Chiểu Đa Kao, Quận 1, Thành Phố Hồ Chí Minh'
    # test = '99 Trần Bình, Mỹ Đình 2 Nam Từ Liêm, Hà Nội'
    corrector = AddressCorrector("srcc/models/address_corrector/vnaddress_trie.pkl")
    # #%%
    # #%%
    print(corrector.correct(test, parse_street=True))
    # print(corrector.correct(test, parse_street=False))

# %%
