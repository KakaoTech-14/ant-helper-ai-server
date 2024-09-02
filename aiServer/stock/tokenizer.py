from konlpy.tag import Okt

okt = Okt()


def okt_tokenizer(text):
    tokens = okt.morphs(text)
    return tokens