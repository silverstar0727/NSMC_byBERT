# NSMC_byKoBERT
Naver Sentiment Movie Corpus Classification를 KoBERT로 진행하였다.

preprocessing과 modeling을 중점으로 진행하였다.

### preprocess.py
preprocessing은 transformers 라이브러리의 BERTTokenizer와 Huggingface의 encode_plus 기능을 이용하여 토크나이징 하였다.
> Huggingface의 Tokenizer(https://github.com/huggingface/tokenizers)

encode_plus의 변환 순서는 아래와 같다
1. 문장을 토크나이징
2. add_special_tokens = True 이면 CLS와 SEP토큰을 붙임
3. 토큰을 인덱싱
4. max_length = MAX_LEN 이면 MAX_LEN으로 길이를 맞춰줌
5. return_attention_mask = True 이면 어텐션 마스크를 생성함
6. 문장이 한개일 경우 토큰 타입은 0으로, 2개일 경우 0과 1로 구분하여 생성

### modeling.py
transformers에서 기본으로 제공하는 pre-trained된 Bert model을 사용하였다.

모델 구성: bert -> dropout -> Dense_layer

