import matplotlib.pyplot as plt
import numpy as np 
from scipy.ndimage import convolve


#Load a sample grey Scale Image

image = np.random.rand(10,10)


#Define Concolution Kernels (Filters)

edge_detection_kernel = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])

blur_kernel = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
])/9


#Apply Convolution

edge_detected_image = convolve(image,edge_detection_kernel)

blurred_image = convolve(image, blur_kernel)

#Visualize the original and Filtered Image

# fig,axes = plt.subplots(1,3,figsize = (12,4)) #1 = Row , 3 = Columns
# axes[0].imshow(image, cmap ='grey')
# axes[0].set_title("Original Image")
# axes[1].imshow(edge_detected_image, cmap ='grey')
# axes[1].set_title("Edge Detection")
# axes[2].imshow(blurred_image, cmap ='grey')
# axes[2].set_title("Blurred image")
# plt.show()


import tensorflow as tf

#Create a Sample input tensor (batch_size,height,width,channels)

image_tensor = tf.random.normal([1,10,10,1])

#Define the Convolutional layer

conv_layer =tf.keras.layers.Conv2D(
    filters=2,
    kernel_size=(3,3),
    strides=(1,1),
    padding='same'

)

#Applying the Convolution

output_tensor = conv_layer(image_tensor)

print(f"Original image : {image_tensor.shape}")
print(f"Output Image: {output_tensor.shape}")


#Achiving the Same thing using Torch
import torch
import torch.nn as nn

#Create a Sample input Tensor(batch_size,channels,height,width)

image_tensor_pt= torch.randn(1,1,10,10)

#Define a Convolutional layer

conv_layer_pt = nn.Conv2d(
    in_channels=1,
    out_channels = 1,
    kernel_size = 3,
    stride =1,
    padding= 1
)

#Apply the convolution

output_tensor_pt = conv_layer_pt(image_tensor_pt)

print(f" Original Shape : {image_tensor_pt.shape}")
print(f"Output Shape : {output_tensor_pt.shape}")