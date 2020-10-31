# transformer를 쓰기 위해서
# pip install transformer
# 위 명령어를 통해서 사전에 설치를 해줘야 함
from transformer import *
import re
import os

import numpy as np
import pandas as pd

# 각종 사전 정보 정의
MAX_LEN = 39
DATA_IN_PATH = 'data_in/KOR'
DATA_OUT_PATH = "data_out/KOR"

# 네이버 영화데이터 불러오기
DATA_TRAIN_PATH = os.path.join(DATA_IN_PATH, "naver_movie", "ratings_train.txt")
DATA_TEST_PATH = os.path.join(DATA_IN_PATH, "naver_movie", "ratings_test.txt")

train_data = pd.read_csv(DATA_TRAIN_PATH, header = 0, delimiter = '\t', quoting = 3)
train_data = train_data.dropna()

# Hugging face의 berttokenizer중 encode_plus기능을 이용하여 tokenize
# 결과값은 dictionary형태로 나옴
# encoded_dict에 저장하여 각각의 key를 이용하여 데이터를 input_id, attention_mask, token_type_id에 저장함
def bert_tokenizer(sent, MAX_LEN):
    encoded_dict = tokenizer.encode_plus(text = sent,
                                         add_special_tokens = True,
                                         max_length = MAX_LEN,
                                         pad_to_max_len = True,
                                         return_attention_mask = True)

    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']

    return input_id, attention_mask, token_type_id


# 각 문장들을 전처리 후 저장할 리스트를 지정
input_ids = []
attention_masks = []
token_type_ids = []
train_data_labels = []


# clean_text함수를 이용하여 특수기호들을 전처리 함
def clean_text(sent):
    sent_clean = re.sub("[^가-힣 ㄱ-ㅎ ㅏ-ㅣ \\s]", " ", sent)
    return sent_clean

# 각각의 문장들을 꺼내어 clean_text를 거친후 bert_tokenizer를 통해 토크나이징 하여 각각의 리스트에 저장함
for train_sent, train_label in zip(train_data['document'], train_data['label']):
    try:
        input_id, attention_mask, token_type_id = bert_tokenizer(clean_text(train_sent), MAX_LEN)
        
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        train_data_labels.append(train_label)
        
    except Exception as e:
        print(e)
        print(train_sent)
        pass
    
# numpy형태로 저장
train_movie_input_ids = np.array(input_ids, dtype = int)
train_movie_attention_masks = np.array(attention_masks, dtype = int)
train_movie_type_ids = np.array(token_type_ids, dtype = int)

# 모든 값들을 함침(preprocessing의 최종단계)
train_movie_inputs = np.array(train_movie_input_ids, train_movie_attention_masks, train_movie_type_ids)
train_data_labels = np.asarray(train_data_labels, dtype = np.int32) # label 값