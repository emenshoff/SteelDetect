from dataset import train
import matplotlib.pyplot as plt
import numpy as np


#Выводим график % распределения классов в размеченных данных
plt.figure(figsize=(13.5,2.5))
bar = plt.bar([1,2,3,4], 100 * np.mean(train.iloc[:,1:5] != '', axis=0))
plt.title('Процентное соотношение дефектов на размеченных данных', fontsize=16)
plt.ylabel('% картинок'); plt.xlabel('Класс дефекта')
plt.xticks([1,2,3,4])
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, '%.1f %%' % height,
             ha='center', va='bottom',fontsize=16)
plt.ylim((0,50)); plt.show()
