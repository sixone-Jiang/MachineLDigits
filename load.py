from sklearn.datasets import load_digits
import numpy as np
digits = load_digits()
# 1.归一化MinMaxScaler()
from sklearn.preprocessing import MinMaxScaler
X_data = digits.data.astype(np.float32)
scaler = MinMaxScaler()
X_data = scaler.fit_transform(X_data)
# 转化为图片的格式
X = X_data.reshape(-1,8,8,1)

import tensorflow
from tensorflow import keras

model = keras.models.load_model('model.h5')
model.predict(X)