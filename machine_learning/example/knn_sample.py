# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:12:43 2019

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

data = pd.read_csv("../data/vehicle.csv")
feature = np.array(data.iloc[:,0:2])
label = data['label'].tolist()

x = np.array(np.arange(data.shape[0]))
y = data.iloc[:,0]

def ShowOriginDataByLabels():
    plt.title('Width & Length of different vehicles')
    plt.scatter(data['length'][data['label']=="car"],data['width'][data['label']=='car'],c='b',label="car")
    plt.scatter(data['length'][data['label']=="truck"],data['width'][data['label']=='truck'],c='r',label="truck")
    plt.legend()

def Euclidean(x1,x2):
    distance = np.sqrt(np.dot((x1-x2),np.transpose(x1-x2)))
    return distance
def CalDistance(x1):
    x1 = np.tile(x1,(data.shape[0],1))
    diff = x1 - feature
    diff = np.sum(diff**2,axis=1)
    diff = diff**0.5
    return diff
def CalDistance2(x1):
    diff=[];
    for index in range(feature.shape[0]):
        x2 = feature[index,:]
        plc = Euclidean(x1,x2)
        diff.append(plc)
    return diff

def SortData(x1):
    data = np.argsort(x1)
    return data

def Labels(k):
    label_count =[]
    classCount={}
    for i in range(k):
        voteLabel = label[sortIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
        label_count.append(voteLabel)
    return label_count,classCount
def MostValue(x1):
    word_counts =Counter(x1)
    top = word_counts.most_common(1)
    return top
test = [4.7,2.1]
#1.计算距离
test_return = np.array(CalDistance(test))
test_return2 = np.array(CalDistance2(test))
#2.排序
sortIndex = SortData(test_return2)
#3.设置K值对其进行统计
label_cout,classCount = Labels(9)

#.获取推测值
top =MostValue(label_cout)

print(test_return)
print(test_return2)
#label = data.iloc[:,2]


#sns.scatterplot(x,y,hue=label)
#label = list(set(label))
#plt.scatter(x,y,c=label)