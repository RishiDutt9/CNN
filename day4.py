from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D, Flatten,Dense,Dropout
import matplotlib.pyplot as plt

#Load CIFER10 dataset

(X_train,y_train),(X_test,y_test)  = cifar10.load_data()

#Normalize the dataset

X_train = X_train.astype('float32') / 255.0 
'''converts to 32 bit flooting number for 
compatability with TF model and normalize the pixel values from 0-255 to 0-1'''
X_test = X_test.astype('float32') / 255.0 

#On hot encoding the labels

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

print(f"Training data shape: {X_train.shape}, Label Shapes : {y_train.shape} ")
print(f"\n Training data shape: {X_test.shape}, Label Shapes : {y_test.shape} ")



#Build the CNN model

model =Sequential([
    Conv2D(32,(3,3),activation ='relu', input_shape = (32,32,3)), #32 = filters ,3,3 = kernal-size
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation = 'relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),#Create a fully connected layer  with 128 units and avtivation
    Dropout(0.5),
    Dense(10 ,activation = 'softmax')
])

model.summary()

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

#train the Model

history =model.fit(
    X_train,y_train,
    epochs = 10,
    batch_size = 64,
    validation_split=0.2
)

#Evaluate on test Datasets
test_loss,test_accuracy= model.evaluate(X_test,y_test)
print(f"Test Accuracy : {test_accuracy:.4f}")


#Plot accuracy 

plt.plot(history.hostory['accuracy'],label = "Training Accuracy" )
plt.plot(history.history['val_accuracy'],label = "Valication Accuracy")
plt.title("Model accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Plot Loss

plt.plot(history.hostory['loss'],label = "Training loss" )
plt.plot(history.history['val_loss'],label = "Valication Loss")
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
