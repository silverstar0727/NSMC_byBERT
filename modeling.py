# tensorflow.__version__==2
import os

import tensorflow as tf
from transformers import *

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#random seed 고정
tf.random.set_seed(1234)
np.random.seed(1234)

BATCH_SIZE = 32
NUM_EPOCHS = 3
VALID_SPLIT = 0.2
MAX_LEN = 39 # EDA에서 추출된 Max Length
DATA_IN_PATH = 'data_in/KOR'
DATA_OUT_PATH = "data_out/KOR"

class TFBertClassifier(tf.keras.Model):
    def __init__(self, dir_path, model_name, num_class):
        super(TFBertClassifier, self).__init__()
        
        self.bert = TFBertModel.from_pretrained(model_name, cache_dir = dir_path)
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_class,
                                                kernel_initializer = tf.keras.initializers.TruncatedNormal(
                                                    self.bert.config.initializer_range),
                                                name = 'classifier')
        
    def call(self, inputs, attention_mask = None, token_type_ids = None, training = False):
        outputs = self.bert(inputs, attention_mask = attention_mask, token_type_ids = token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training = training)
        logits = self.classifier(pooled_output)
        
        return logits

cls_model = TFBertClassifier(model_name = 'bert_base_multilingual-cased',
                             dir_path = 'bert_ckpt',
                             num_class = 2)

optimizer = tf.keras.optimizers.Adam(3e-5)
loss = tf.keras.losses.SarseCategoricalCrossentropy(from_logits = True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

cls_model.compile(optimizer = optimizer, loss = loss, metric = [metric])


model_name = 'naver_sentimment_classifier_model_byBERT'

# overfitting을 막기 위한 earlystop 추가
earlystop_callback = EarlyStopping(monitor = 'val_accuracy', min_delta = 0.0001, patience = 2)
# min_delta: val_accuracy가 0.0001보다 작게 오르면 개선이 없다고 판단
# patience: 개선이 없는 epochs가 2번 이어져도 기다려줌을 의미

checkpoint_path = os.path.join(DATA_OUT_PUT, model_name, 'weights.h5') # 체크포인트를 저장할 위치
checkpoint_dir = os.path.dirname(checkpoint_path) # 저장한 모델 위치를 os.path.dirname으로 불러옴

# Create path if don't exists
if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))
    
cp_callback = ModelCheckpoint(checkpoint_path,
                              monitor = 'val_accuracy',
                              verbose = 1,
                              save_best_only = True,
                              save_weights_only = True)

history = cls_model.fit(train_movie_inputs, 
                        train_data_labels, 
                        epochs = NUM_EPOCHS, 
                        bach_size = BATCH_SIZE,
                        validation_split = VALID_SPLIT,
                        callbacks = [earlystop_callback, cp_callback])

#steps_for_epoch
print(history.history)