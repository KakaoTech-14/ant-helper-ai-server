import joblib
from konlpy.tag import Okt


import os
import pandas as pd

from stock.tokenizer import okt, okt_tokenizer

os.environ["PYTHONIOENCODING"] = "utf-8"

import warnings
warnings.filterwarnings(action='ignore')

nsmc_train_df = pd.read_csv("C:/Users/hl347/Desktop/base_storage_location/project/antHelperAIServer/aiServer/static/ratings_train.txt", encoding='utf8', sep='\t', engine='python')
nsmc_train_df.head()

nsmc_train_df = nsmc_train_df[nsmc_train_df['document'].notnull()]

# 부정 0 긍정 1

nsmc_train_df['label'].value_counts()

import re

nsmc_train_df['document'] = nsmc_train_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))
nsmc_train_df.head()

nsmc_test_df = pd.read_csv("C:/Users/hl347/Desktop/base_storage_location/project/antHelperAIServer/aiServer/static/ratings_test.txt", encoding='utf8', sep='\t', engine='python')
nsmc_test_df.head()

nsmc_test_df = nsmc_test_df[nsmc_test_df['document'].notnull()]

print(nsmc_test_df['label'].value_counts())

nsmc_test_df['document'] = nsmc_test_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', "", x))




# 15분~20분 소요

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
tfidf.fit(nsmc_train_df['document'])
nsmc_train_tfidf = tfidf.transform(nsmc_train_df['document'])
print('*** TF-IDF 기반 피처 벡터 생성 ***')

joblib.dump(tfidf, '../static/tfidf.pkl')
# To load the model later, you would use:
# SA_lr_best = joblib.load('SA_lr_best.pkl')
