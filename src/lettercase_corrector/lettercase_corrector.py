class LettercaseCorrector(object):
    common_wrongcase = ['Ơ', 'Ở', 'Ờ', 'Ợ', 'Ớ',  # Ơ
                     'SƠ', 'SỞ', 'SỜ', 'SỢ', 'SỚ',
                     'CƠ', 'CỜ', 'CỚ', 'CỠ',
                     'VƠ', 'VỠ', 'VỞ', 'VỜ', 'VỢ', 'VỚ',
                     'SO', 'SỌ',
                     'CO', 'CỎ', 'CÓ', 'CÒ', 'CỌ'
                     'VO', 'VỎ', 'VÓ', 'VÒ', 'VÕ',
                     'VỀ', 'VẾ', 'VỆ',
                     'XẾ', 'XỆ',
                     'Ô', 'Ổ', 'Ồ', 'Ố',
                     'SỐ', 'SỔ',
                     'VÔ', 'VỐ', 'VỒ'
                     'CÔ', 'CỐ', 'CỔ', 'CỖ', 'CỘ',
                     'SỌC', 'SÓC',
                     'VỌC', 'VÓC',
                     'ƠI', 'ỚI',
                     'OS'
                     'ƯU',
                     'CỨU', 'CỬU', 'CỪU',
                     'SỬU',
                     'U', 'Ú', 'Ù', 'Ủ', 'Ụ'
                     'XU', 'XÚ', 'XÙ', 'XỤ'
                     'CU', 'CÚ', 'CÙ', 'CỦ', 'CŨ', 'CỤ',
                     'SU', 'SÚ',
                     'VU', 'VÚ', 'VÙ', 'VŨ', 'VỤ',
                     'ƯỚC'
                     ]
    common_wrongcase.extend([key.lower() for key in common_wrongcase])
    @staticmethod
    def correct(text:str):
        lwords = text.split(" ")
        for i in range(2,len(lwords)-2):
            if lwords[i] in LettercaseCorrector.common_wrongcase:
                #correct random uppercase
                if not lwords[i-2].islower() and not lwords[i-1].islower() and lwords[i+1].isupper() and lwords[i+2].isupper() and lwords[i].islower(): # TÔI CÓ cơ SỞ ĐỂ
                    lwords[i] = lwords[i].upper()
                #correct random lowercase
                elif not lwords[i-2].isupper() and  not lwords[i-1].isupper() and lwords[i+1].islower() and lwords[i+2].islower() and lwords[i].isupper(): #tôi có CƠ sở để
                    lwords[i] = lwords[i].lower()
                elif lwords[i] =='CƠ' and lwords[i+1]=='SỞ':
                    if not lwords[i-1].isupper() and lwords[i-2].islower() and lwords[i+2].islower():
                        lwords[i] = 'cơ'
                        lwords[i+1] = 'sở'
                elif lwords[i] =='cơ' and lwords[i+1]=='sở':
                    if not lwords[i-1].islower() and lwords[i-2].isupper() and lwords[i+2].isupper():
                        lwords[i] = 'CƠ'
                        lwords[i+1] = 'SỞ'
                # #exclusive case when before consỉdering word is end of sentence -> #not necesssarily, error analysis show there is no such case
                # if lwords[i-1] not in '.?!':
                #     if lwords[i+1].isupper and lwords[i+1].isupper():
                        # lwords[i] = lwords[i].title()
        if lwords[0] in LettercaseCorrector.common_wrongcase:
            if lwords[1].isupper() and lwords[2].isupper() and  lwords[0].islower():
                lwords[0] = lwords[0].upper()
            if lwords[1].islower() and lwords[2].islower() and  lwords[0].isupper():
                lwords[0] = lwords[0].lower()
        if lwords[1] in LettercaseCorrector.common_wrongcase:
            if not lwords[0].islower() and lwords[2].isupper() and lwords[3].isupper() and  lwords[1].islower():
                lwords[1] = lwords[1].upper()
            if not lwords[0].isupper() and lwords[2].islower()  and lwords[3].islower() and  lwords[1].isupper():
                lwords[1] = lwords[1].lower()
        if lwords[-1] in LettercaseCorrector.common_wrongcase:
            if lwords[-2].isupper() and lwords[-3].isupper() and lwords[-1].islower():
                lwords[-1] = lwords[-1].upper()
            if lwords[-2].islower() and lwords[-3].islower() and lwords[-1].isupper():
                lwords[-1] = lwords[-1].lower()   
        if lwords[-2] in LettercaseCorrector.common_wrongcase:
            if lwords[-1].isupper() and lwords[-3].isupper() and lwords[-4].isupper() and lwords[-2].islower():
                lwords[-2] = lwords[-2].upper()
            if lwords[-1].islower() and lwords[-3].islower()  and lwords[-4].islower() and lwords[-2].isupper():
                lwords[-2] = lwords[-2].lower()
        return " ".join(lwords)
    

#%%
if __name__ == "__main__":
    #%%
    texts = ["tôi có một CƠ SỞ sản xuất bún đậu mắm tôm", "I love you SO much but you don't know"]
    for text in texts:
        print("Before: ", text)
        print("After: ", LettercaseCorrector.correct(text))
