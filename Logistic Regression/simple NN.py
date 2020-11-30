import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Dropout,Flatten
from keras.models import Sequential
data=pd.read_csv('labeled_spotify_data_genre_clean_balanced.csv')
x=data.drop('Label',axis=1)
y=data['Label']
x=x.astype('float')
x=(x-x.min())/(x.max()-x.min())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=Sequential()
model.add(Dense(units=128,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=512,activation='tanh'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print('Train...')
model.fit(x_train,y_train,batch_size=10,epochs=100,validation_split=0.1)
score,acc=model.evaluate(x_test, y_test,batch_size=10)
print('Test score:',score)
print('Test accuracy:',acc)