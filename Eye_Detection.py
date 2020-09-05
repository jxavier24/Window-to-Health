import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm

eyes_file=open('eyes','rb')
data=np.load(eyes_file)
labels=np.load(eyes_file)

test_data=np.load(eyes_file)
test_labels=np.load(eyes_file)

class_names=['not healthy','healthy']

model= keras.Sequential([
    keras.layers.Flatten(input_shape=(128,128)), 
    keras.layers.Dense(128,activation='relu'), 
    keras.layers.Dense(2,activation='softmax') 
])

model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(data,labels,epochs=100) 

test_loss,test_acc= model.evaluate(test_data, test_labels)

print('Tested Acc',test_acc)

prediction= model.predict(test_data)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_data[i])
    plt.xlabel('Actual:'+ class_names[test_labels[i]])
    plt.title("Prediction:"+class_names[np.argmax(prediction[i])])
    plt.show()
