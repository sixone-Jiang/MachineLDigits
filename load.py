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

# ----

# 2.独热编码oneHot
from sklearn.preprocessing import OneHotEncoder
# y = digits.target.reshape(-1,1)
# 将Y_data变为一列
y = digits.target.astype(np.float32).reshape(-1,1) 
Y = OneHotEncoder().fit_transform(y).todense()

# 3.切分数据集train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0,stratify=Y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten
# ------
import numpy as np
import matplotlib.image as mpimg # mpimg 用于读取图片
def create_model():
    # 4.建立模型
    model = Sequential()
    ks = (3, 3)  # 卷积核的大小
    input_shape = X_train.shape[1:]
    # 一层卷积，padding='same',tensorflow会对输入自动补0
    model.add(Conv2D(filters=16, kernel_size=ks, padding='same', input_shape=input_shape, activation='relu'))
    # 池化层1
    model.add(MaxPool2D(pool_size=(2, 2)))
    # 防止过拟合，随机丢掉连接
    model.add(Dropout(0.25))
    # 二层卷积
    model.add(Conv2D(filters=32, kernel_size=ks, padding='same', activation='relu'))
    # 池化层2
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 三层卷积
    model.add(Conv2D(filters=64, kernel_size=ks, padding='same', activation='relu'))
    # 四层卷积
    model.add(Conv2D(filters=128, kernel_size=ks, padding='same', activation='relu'))
    # 池化层3
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 平坦层
    model.add(Flatten())
    # 全连接层
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    # 激活函数softmax
    model.add(Dense(10, activation='softmax'))
    # ----训练
    # 模型训练
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()
model.load_weights('model_weights.h5')
image = mpimg.imread('16.jpg') 
# 此时 image 就已经是一个 np.array 了，可以对它进行任意处理
model.predict(image)