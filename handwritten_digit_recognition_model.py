import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# Resize images for GoogLeNet
x_train = tf.image.resize(x_train,(96,96))
x_test = tf.image.resize(x_test,(96,96))

# Convert grayscale to RGB
x_train = tf.image.grayscale_to_rgb(x_train)
x_test = tf.image.grayscale_to_rgb(x_test)

# Inception block
def inception_module(x, f1, f3r, f3, f5r, f5, proj):

    path1 = Conv2D(f1,(1,1),padding='same',activation='relu')(x)

    path2 = Conv2D(f3r,(1,1),padding='same',activation='relu')(x)
    path2 = Conv2D(f3,(3,3),padding='same',activation='relu')(path2)

    path3 = Conv2D(f5r,(1,1),padding='same',activation='relu')(x)
    path3 = Conv2D(f5,(5,5),padding='same',activation='relu')(path3)

    path4 = MaxPooling2D((3,3),strides=(1,1),padding='same')(x)
    path4 = Conv2D(proj,(1,1),padding='same',activation='relu')(path4)

    return Concatenate()([path1,path2,path3,path4])

# Input layer
input_layer = Input(shape=(96,96,3))

x = Conv2D(64,(7,7),strides=2,padding='same',activation='relu')(input_layer)
x = MaxPooling2D((3,3),strides=2,padding='same')(x)

x = inception_module(x,64,96,128,16,32,32)
x = inception_module(x,128,128,192,32,96,64)

x = GlobalAveragePooling2D()(x)

output = Dense(10,activation='softmax')(x)

model = Model(input_layer,output)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(x_train,y_train,epochs=9,batch_size=64)

# Evaluate
test_loss,test_acc = model.evaluate(x_test,y_test)
print("Test Accuracy:",test_acc)

# Show images
plt.figure(figsize=(8,8))

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(x_test[i])
    plt.title("Label: "+str(y_test[i]))
    plt.axis('off')

plt.show()