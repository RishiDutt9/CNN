import matplotlib.pyplot as plt
from torchvision import datasets,transforms

#Load the Data

transform = transforms.ToTensor()
train_dataset = datasets.CIFAR10(root='./data',train= True,transform = transform,download = True)

#Visualize the Sample Images

figs,axes = plt.subplots(1,5,figsize=(12,3))

for i in range(5):
    image,label = train_dataset[i]
    axes[i].imshow(image.permute(1,2,0))
    axes[i].axis('off')
    axes[i].set_title(f"Label :{label}")
plt.show()

#Display Pixel values for first image
# image.label = train_dataset[0]
# print(f"Image Shape {image.shape}")
# print(f"Pixel Values: ")
# print(image)


import tensorflow as tf

# Define a Basic TensorFlow Model

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters = 32,kernel_size = (3,3), activation= 'relu', input_shape = (32,32,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation = 'relu'),
    tf.keras.layers.Dense(10,activation = 'softmax'),
])

#Compile the model

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

print("TensorFlow CNN model is Ready")

import torch.nn as nn

#Define simple CNN

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size = 3,activation = 'relu')
        self.pool = nn.maxPool2d(2,2)
        self.fc1 = nn.Linear(32*15*15,128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self,x):
        x = relu(self.conv1(x))
        x= self.pool(x)
        x = x.view(-1,32*15*15)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

print("Torch CNN is ready")


