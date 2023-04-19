from src.datetime_corrector.datetime_corrector import DatetimeCorrector
from src.lettercase_corrector.lettercase_corrector import LettercaseCorrector
from src.address_corrector.address_corrector import AddressCorrector, Trie, TrieNode
from src.ocr_corrector.ocr_corrector import Predictor as OcrCorrector
ocr_corrector_cfg = {'device':'cpu', 'model_type':'seq2seq', 'weight_path':'weights/seq2seq_1.pth'}
address_corrector_cfg = {"model_path":"src/address_corrector/vnaddress_trie.pkl"}
class Corrector:
    def __init__(self, kwargs_address:dict=address_corrector_cfg, kwargs_ocr:dict=ocr_corrector_cfg) -> None:
        self._address_corrector = AddressCorrector(**kwargs_address)
        self._ocr_corrector = OcrCorrector(**kwargs_ocr)
        pass
    def __call__(self, text, mode):
        if mode == "datetime":
            return DatetimeCorrector.correct(text)
        elif mode =="lettercase":
            return LettercaseCorrector.correct(text)
        elif mode=="address":
            return self._address_corrector.correct(text)
        elif mode=="ocr":
            return self._ocr_corrector.predict(text)
        else:
            raise NotImplementedError("Unsupported mode: ", mode)


if __name__ == "__main__":
    corrector = Corrector()
    print(corrector("ngày /date 01 tháng /month 04 năm/year✪2022", "datetime"))
    print(corrector('16.5 C/C 4 Nguyễn Đinh Chieu Đa Kao, Quận 1, TP HoChí Minh', "address"))
    print(corrector('tôi có một CƠ SỞ sản xuất bún đậu mắm tôm', "lettercase"))
    print(corrector("Sau khi có y kien của Phó thủ tướng Trần Hồng Hà, UBND tinh Đồng Nai đồng ý gia han 4 mỏ đất phục vụ đắp nền cho tuyen cao tốc chạy qua địa bàn", "ocr"))