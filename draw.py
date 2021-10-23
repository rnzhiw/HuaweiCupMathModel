import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot(matrix):
  sns.set()
  f,ax=plt.subplots()
  print(matrix) #打印出来看看
  sns.heatmap(matrix, annot=True,
              xticklabels=['Small', 'Fit', 'Large'],
              yticklabels=['Small', 'Fit', 'Large'],
              cmap="Blues", ax=ax, fmt='.20g') #画热力图
  ax.set_title('Confusion Matrix') #标题
  ax.set_xlabel('Predict') #x轴
  ax.set_ylabel('True') #y轴

matrix=np.array([[1116, 587, 105],
 [827, 10134, 855],
 [66, 355, 955]])
plot(matrix)# 画原始的数据
plt.show()
