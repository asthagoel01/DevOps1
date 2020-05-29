#!/usr/bin/env python
# coding: utf-8

# In[3]:


import keras
from keras.datasets import mnist
import numpy as np


# In[4]:


dataset=mnist.load_data('mymnist.db')


# In[5]:


train,test=dataset
x_train,y_train=train
x_test,y_test=test


# In[6]:


x_train=x_train.reshape((x_train.shape[0],28,28,1))
x_test=x_test.reshape((x_test.shape[0],28,28,1))


# In[7]:




x_train2=(x_train/255)-0.5
x_test2=(x_test/255)-0.5


# In[8]:




NUM_CLASSES=10
y_train2=keras.utils.to_categorical(y_train,10)
y_test2=keras.utils.to_categorical(y_test,10)


# In[9]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam,RMSprop,Adadelta
def make_model():
    model=Sequential()
    model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(28,28,1)))
    model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=None,padding='valid',data_format=None))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256,input_shape=(256, )))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    return model


# In[10]:



model=make_model()
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])



# In[11]:


from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("mnist.h5",monitor="val_loss",mode="min",save_best_only = True,verbose=1)
earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3,verbose = 1,restore_best_weights = True)
# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]
BATCH_SIZE=32
EPOCHS=2
model.fit(x_train2,y_train2, batch_size=BATCH_SIZE,epochs = EPOCHS,validation_data = (x_test2,y_test2),callbacks=callbacks,shuffle=True,verbose=1)


# In[12]:



scores=model.evaluate(x_test2,y_test2,verbose=1)
print('Test Loss:',scores[0])
print('Test Accuracy:',scores[1])


# In[ ]:




