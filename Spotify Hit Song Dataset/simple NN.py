import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout,Flatten
from keras.models import Sequential
data=pd.read_csv('dataset-of-10s.csv')
x=data.drop(columns=['target','track','artist','uri','chorus_hit','sections'])
x=x.astype('float')
x=(x-x.min())/(x.max()-x.min())
y=data['target']

bias=[]
for i in range(0,data.shape[0]):
    bias.append(1)
x['bias']=bias

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model=Sequential()
model.add(Dense(units=128,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print('Train...')
model.fit(x_train,y_train,batch_size=10,epochs=20,validation_split=0.1)
score,acc=model.evaluate(x_test, y_test,batch_size=10)
print('Test score:',score)
print('Test accuracy:',acc)
model.save('Neural Network')