import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import random
data=pd.read_csv('labeled_spotify_data_genre_clean_balanced.csv')
x=data.drop(columns=['Label'])
x=x.astype('float')
x=(x-x.min())/(x.max()-x.min())
y=data['Label']


bias=[]
for i in range(0,data.shape[0]):
    bias.append(1)
x['bias']=bias

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

theta=[]
theta_temp=[]
for i in range(0,14):
    theta.append(1)



def sigmoid(theta,x):
    theta=np.mat(theta)
    x=np.mat(x)
    first=theta.dot(x.T)
    second=float(first)
    result=1/(1+np.exp(-second))
    return result

def gradient_descent(eta,x,y,theta):
    first=sigmoid(theta,x)
    second=first-int(y)
    third=eta*second*np.mat(x)
    result=theta-third
    return result

for k in range(0,100000):
    j=random.randint(0,x_train.shape[0]-1)
    theta=gradient_descent(0.0001,x_train.iloc[j],y_train.iloc[j],theta)
print(theta)

TP=0
TN=0
FP=0
FN=0
total=0

for i in range(0,x_test.shape[0]):
    result=sigmoid(theta,x_test.iloc[i])
    if result>=0.5:
        result=1
    else:
        result=0
    if result==1 and result==y_test.iloc[i]:
        TP+=1
    if result==0 and result==y_test.iloc[i]:
        TN+=1
    if result==1 and result!=y_test.iloc[i]:
        FP+=1
    if result==0 and result!=y_test.iloc[i]:
        FN+=1
    total+=1

precision=TP/(TP+FP)
recall=TP/(TP+FN)
print('test set:')
print('accuracy:')
print((TP+TN)/total)
print('precision:')
print(precision)
print('recall:')
print(recall)
print('F1')
print(2*precision*recall/(precision+recall))
        
TP=0
TN=0
FP=0
FN=0
total=0

for i in range(0,x_train.shape[0]):
    result=sigmoid(theta,x_train.iloc[i])
    if result>=0.5:
        result=1
    else:
        result=0
    if result==1 and result==y_train.iloc[i]:
        TP+=1
    if result==0 and result==y_train.iloc[i]:
        TN+=1
    if result==1 and result!=y_train.iloc[i]:
        FP+=1
    if result==0 and result!=y_train.iloc[i]:
        FN+=1
    total+=1

precision=TP/(TP+FP)
recall=TP/(TP+FN)
print('training set:')
print('accuracy:')
print((TP+TN)/total)
print('precision:')
print(precision)
print('recall:')
print(recall)
print('F1')
print(2*precision*recall/(precision+recall))

