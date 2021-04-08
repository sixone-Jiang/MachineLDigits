# 使用此程序导出标准数据集
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
digits = datasets.load_digits()
print(len(digits.images))
i = 0
s = ''
for image, label in zip(digits.images, digits.target):
    i = i + 1
    plt.imshow(image)
    plt.savefig('./data/' + str(i) + '.jpg')
    s = s + str(label)

with open("yanzheng.txt", "w", encoding='utf-8') as f:
    f.write(s)
    f.close()

