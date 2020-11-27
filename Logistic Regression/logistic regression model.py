import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

data=pd.read_csv('labeled_spotify_data_genre_clean.csv')
x=data.drop(columns=['Label'])
x=x.astype('float')
x['duration_s']=x['duration_ms']/1000
x=x.drop(columns='duration_ms')
y=data['Label']

bias=[]
for i in range(0,data.shape[0]):
    bias.append(1)
x['bias']=bias

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

theta=[]
for i in range(0,14):
    theta.append(0)


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



for j in range(0,x_train.shape[0]):
    theta=gradient_descent(0.00001,x_train.iloc[j],y_train.iloc[j],theta)
print(theta)

correct=0
total=0
for i in range(0,x_test.shape[0]):
    result=sigmoid(theta,x_test.iloc[i])
    if result>=0.5:
        result=1
    else:
        result=0
    if result==y_test.iloc[i]:
        correct+=1
    total+=1

print(correct/total)
        

