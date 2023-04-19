# %%
from pathlib import Path
import json
import unicodedata
import pickle
DATA_DIR = Path('data/vietnam_dataset')
# key should be lowercase and not contain reduntant space
SBIG_CITIES = {"cần thơ", "hà nội", "hồ chí minh", "đà nẵng", "hải phòng"}
D_UPDATE_LEVEL = {"thủ đức": "thành phố"}  # key should be lowercase and not contain reduntant space
# %%


def preprocess(text):
    # text = unicodedata.normalize('NFC', text) #vnaddress data have to be normalize to unicode
    text = unicodedata.normalize('NFC', text)  # vnaddress data have to be normalize to unicode
    return text


class TrieNode:
    def __init__(self):
        self.word = None
        self.children = {}

    def insert(self, word):
        node = self
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()
            node = node.children[letter]
        node.word = word


class Trie():
    def __init__(self):
        self.root = TrieNode()

    def update(self, ltexts):
        for text in ltexts:
            self.root.insert(preprocess(text))

    def __searchRecursive(self, node, letter, word, previousRow, results, maxCost):
        # http://stevehanov.ca/blog/?id=114
        # This recursive helper is used by the search function above. It assumes that
        # the previousRow has been filled in already.
        columns = len(word) + 1
        currentRow = [previousRow[0] + 1]
        # Build one row for the letter, with a column for each letter in the target
        # word, plus one for the empty string at column 0
        for column in range(1, columns):
            insertCost = currentRow[column - 1] + 1
            deleteCost = previousRow[column] + 1

            if word[column - 1] != letter:
                replaceCost = previousRow[column - 1] + 1
            else:
                replaceCost = previousRow[column - 1]
            currentRow.append(min(insertCost, deleteCost, replaceCost))
        # if the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a word in this trie node, then add it.
        if currentRow[-1] <= maxCost and node.word != None:
            results.append((node.word, currentRow[-1]))
        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if min(currentRow) <= maxCost:
            for letter in node.children:
                self.__searchRecursive(node.children[letter], letter, word, currentRow,
                                       results, maxCost)

    def search(self, word, thresh=0.35):
        # build first row
        currentRow = range(len(word) + 1)
        max_cost = int(len(word) * thresh)  # number of character to wrong
        results = []
        # recursively search each branch of the trie
        for letter in self.root.children:
            self.__searchRecursive(self.root.children[letter], letter, word, currentRow,
                                   results, max_cost)
        results.sort(key=lambda x: x[1])  # highest score on top
        return results


def load_vn_dataset():
    ldata = []
    for file_path in DATA_DIR.glob('*.json'):
        # if file_path.name!="HD.json":
        #     continue #TODO: remove this
        with open(file_path, 'rb') as f:
            data = json.load(f)
        ldata.append(data)
    return ldata


def convert_ldata_to_ddata(ldata):
    ddata = {}
    for data in ldata:
        ddata[data['name']] = data
    return ddata


def convert_ddata_to_ltexts(ddata):
    return [k.lower().strip() for k in ddata]


def build_trie_from_data(data):
    trie_ = Trie()
    if not data:
        return trie_
    data = convert_ddata_to_ltexts(convert_ldata_to_ddata(data)) if isinstance(data[0], dict) else data
    trie_.update(data)
    return trie_


def main():
    ldata_pro = load_vn_dataset()
    ltexts_pro = convert_ddata_to_ltexts(convert_ldata_to_ddata(ldata_pro))
    trie_pro = Trie()
    trie_pro.update(ltexts_pro)
    dict_save = {"trie": trie_pro}
    for data_pro in ldata_pro:
        pro_name = preprocess(data_pro["name"].lower().strip())
        trie_dis = build_trie_from_data(data_pro["district"])
        dict_save[pro_name] = {"trie": trie_dis,
                               "pre": "tỉnh" if pro_name not in SBIG_CITIES else "thành phố"}  # hard code warning
        for data_dis in data_pro["district"]:
            dis_name = preprocess(data_dis["name"].lower().strip())
            dis_pre = preprocess(data_dis["pre"].lower().strip())
            trie_ward = build_trie_from_data(data_dis["ward"])
            trie_street = build_trie_from_data(data_dis["street"])
            dict_save[pro_name][dis_name] = {
                "pre": dis_pre if dis_name not in D_UPDATE_LEVEL else D_UPDATE_LEVEL[dis_name],
                "trie_ward": trie_ward,
                "trie_street": trie_street}
            for data_ward in data_dis["ward"]:
                ward_name = preprocess(data_ward["name"].lower().strip())
                ward_pre = preprocess(data_ward["pre"].lower().strip())
                dict_save[pro_name][dis_name][ward_name] = {"pre": ward_pre}  # TODO: D_UPDATE_LEVEL?

    with open('vnaddress_trie.pkl', 'wb') as f:
        pickle.dump(dict_save, f)


#     def __init__(self):
#         self.trie_dict = {}
#         self.tree_dict = {}
#     def construct_trie_dict(self):
#         ldata = load_vn_dataset()
#         self.trie_dict['Việt Nam'] = Trie()
#         self.trie_dict['Việt Nam'].update(convert_ldata_to_ltexts(ldata))
#         self.tree_dict['Việt Nam'] = {}
#         for province in ldata:
#             for
# %%
if __name__ == "__main__":
    main()
    # %%
    import pickle
    with open('vnaddress_trie.pkl', 'rb') as f:
        vnaddress_trie_dict = pickle.load(f)
    # %%
    hn = vnaddress_trie_dict["trie"].search('hà nội', 1)[0][0]

    # %%
    assert hn == "hà nội", "the data is not normalized yet, please normalize with NFC formatp"
    # %%
    unicodedata.normalize('NFC', hn) == "hà nội"
    # #%%
    # # trie = Trie()
    # pass

    # test = "Aa Điền X Cộng Hòa, H. Nam Sách, T. Hai Dương"

    # #%%
    # ldata = load_vn_dataset()
    # ddata = convert_ldata_to_ddata(ldata)
    # ltexts = convert_ddata_to_ltexts(ddata)
    # #%%
    # ldata
    # # %%
    # trie = Trie()
    # trie.update(ltexts)
    # #$$
    # #%%
    # trie.search('Ha Noi', 0.5)
    # # %%
    # trie.search('thừa thiên huế', 0.35)
    # #%%
    # ldata_dis_hd = ddata['Hải Dương']['district']
    # trie_dis_hd = Trie()
    # trie_dis_hd.update(convert_ddata_to_ltexts(convert_ldata_to_ddata(ldata_dis_hd)))
    # #%%

    # ldata_ward_hd
    # #%%
    # ldata_street_hd = convert_ldata_to_ddata(ddata['Hải Dương']['district'])["Nam Sách"]["street"]
    # trie_street_hd = Trie()
    # trie_street_hd.update(ldata_street_hd)
    # ldata_street_hd

    # #%%
    # from underthesea import word_tokenize
    # lsent = test.split(",")
    # print(lsent)

    # def trie_look_up(trie, word, threshold = 0.8):
    #     try:
    #         parsed_address_, edit_distance = trie.search(word)[0]
    #     except IndexError:
    #         return word
    #     return parsed_address_ if 1-edit_distance/len(lwords[-1]) > threshold else word

    # ltrie = [trie, trie_dis_hd, trie_ward_hd, trie_street_hd]
    # lparrsed_address = []
    # for i, sent in enumerate(lsent[::-1]):
    #     lwords = word_tokenize(sent)
    #     lparrsed_address.append(trie_look_up(ltrie[i], lwords[-1]))
    #     if len(lwords) ==3:
    #         lparrsed_address.append(trie_look_up(ltrie[i+1], lwords[0]))
    # print(lparrsed_address)

    ###########################################################################################
    # %%
    # ldata_dis_hcm = ddata['Hồ Chí Minh']['district']
    # ldata_dis_hcm

    # #%%

    # #%%
    # convert_ddata_to_ltexts(convert_ldata_to_ddata(ldata_dis_hcm))

    # #%%
    # trie_dis_hcm = Trie()
    # trie_dis_hcm.update(convert_ddata_to_ltexts(convert_ldata_to_ddata(ldata_dis_hcm)))

    # #%%
    # res = trie_dis_hcm.search('Quận 3')
    # res
    # #%%
    # ldata_ward_q3_hcm =  convert_ldata_to_ddata(ddata['Hồ Chí Minh']['district'])['Quận 3']['ward']
    # ldata_ward_q3_hcm
    # #%%
    # convert_ddata_to_ltexts(convert_ldata_to_ddata(ldata_ward_q3_hcm))
    # #%%
    # trie_ward_q3_hcm = Trie()
    # trie_ward_q3_hcm.update(convert_ddata_to_ltexts(convert_ldata_to_ddata(ldata_ward_q3_hcm)))
    # #%%
    # # trie_ward_q3_hcm.search('Bến Nghỉ')
    # trie_ward_q3_hcm.search('Phường Bến Nghé')
    # # %%
    # ltrie = [trie, trie_dis_hcm, trie_ward_q3_hcm]
    # test = '82 Lý Chính Thắng P.S Quan 3, Thành phố Hồ Chi Minh'
    # test = '82 Lý Chính Thắng P.S Qua 3, Thành pho Hồ Chi Minh'
    # # test = '64 26 27 Nguyễn Khoái P2 04' #failed
    # # ltrie = [trie_dis_hcm, trie_ward_q3_hcm]
    # # test =  '486/6 Phan Xích Long Phường 03 C Phú Nhuân'
    # # test =
    # # ầu 11, Dương flaza 34 Lê Duỡn Bến Nghỉ Quận 1'
    # test = '303 y Thường Việt Phường 15 Xuân 3 TP HCM'
    # test = 'Số2 đường Đăng Như Mai Phường Thanh Mỹ La Thành phố Thủ Đức, Thành phố Hồ Chí Minh'
    # test = ' 7/6/13A Khu Pho 4 Lý Te Xuyên Phương Linh Đông Quận Thu ĐỨC Thành Thơ Hồ Chí Minh Việt Nam'
    # ldata_ward_hcm =  convert_ldata_to_ddata(ddata['Hồ Chí Minh']['district'])['Thủ Đức']['ward']
    # #%%

    # trie_ward_hcm = Trie()
    # trie_ward_hcm.update(convert_ddata_to_ltexts(convert_ldata_to_ddata(ldata_ward_hcm)))
    # ltrie = [trie, trie_dis_hcm, trie_ward_hcm]
    # depth = 0
    # lphrases = test.split(',')
    # correct = []
    # for p in lphrases[::-1]:
    #     lwords = p.split()
    #     cursor = len(lwords)
    #     while (cursor> 0):
    #         flag_matching = False
    #         if depth<len(ltrie):
    #             for i in [3,2]:
    #                 p_ = ' '.join(lwords[max(0,cursor-i):cursor])
    #                 res = ltrie[depth].search(p_, 0.35)
    #                 if res:
    #                     depth +=1
    #                     # print(res)
    #                     cursor -= i
    #                     correct = [res[0][0]] + correct
    #                     flag_matching = True
    #                     break
    #         if not flag_matching:
    #             correct = [lwords[cursor-1]] + correct
    #             cursor -= 1 #else
    #     correct = [','] + correct
    # print(correct[1:])

    # # %%test = '82 Lý Chính Thắng P.S Quan 3, Thành phố Hồ Chi Minh'

    # trie_ward_hcm.search('Thanh Mỹ La')
    # # %%
