import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
# 엑셀 파일 로드

data = pd.read_excel("D://dataset//textEmotions//dataset_v.xlsx", engine='openpyxl')

# 예시로, 텍스트와 레이블 열 이름이 'text'와 'label'이라고 가정

texts = data['text'].astype(str).tolist()

labels = data['label'].astype(int).tolist()

# 텍스트 데이터를 정수 인덱스 시퀀스로 변환

tokenizer = Tokenizer(num_words=5000)  # 5000개의 단어만 사용

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

# 패딩

max_len = 100  # 시퀀스 길이

X = pad_sequences(sequences, maxlen=max_len)

y = pd.get_dummies(labels).values  # 레이블을 원-핫 인코딩

# 학습/검증 데이터 분리

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vocab_size = 5000

embedding_dim = 128

model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))

model.add(LSTM(128, return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(128))

model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))  # 다중 클래스 분류
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 모델 학습
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# 검증 데이터에 대한 성능 평가
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test Accuracy: {accuracy}')

# 입력된 언어
input_text = '행복하다'

# 텍스트를 정수 시퀀스로 변환
input_sequence = tokenizer.texts_to_sequences([input_text])

# 패딩 적용
input_sequence_padded = pad_sequences(input_sequence, maxlen=max_len)

# 모델을 사용하여 예측 수행
predictions = model.predict(input_sequence_padded)

# 예측 결과 해석
predicted_label_index = np.argmax(predictions[0])
print(f"입력된 언어 '{input_text}'는 레이블 '{predicted_label_index}'에 해당합니다.")
