#importing the required libraries
import os
import numpy as np
import tensorflow as tf

#normalizing image pixels to [-1,1]
def normalize(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image

#downsample block of the architecture
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                              kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

#upsample block of the architecture
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

#Pix2Pix generator using u-net inspired architecture
def Generator():
    inputs = tf.keras.layers.Input(shape=[None,None,3])

    down_stack = [
    downsample(64, 4, apply_batchnorm=False), 
    downsample(128, 4), 
    downsample(256, 4), 
    downsample(512, 4), 
    downsample(512, 4), 
    downsample(512, 4), 
    downsample(512, 4), 
    downsample(512, 4), 
  ]

    up_stack = [
    upsample(512, 4, apply_dropout=True), 
    upsample(512, 4, apply_dropout=True), 
    upsample(512, 4, apply_dropout=True), 
    upsample(512, 4), 
    upsample(256, 4), 
    upsample(128, 4), 
    upsample(64, 4), 
  ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') 

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

#helper function to load weights and build the generator
def build_generator():
    generator = Generator()
    CURRENT_WORKING_DIRECTORY = str(os.getcwd())
    generator.load_weights(CURRENT_WORKING_DIRECTORY + '/models/derain.h5')
    return generator

#preprocessing function
def preprocess(image):
    image = np.array(image)
    image = normalize(image)
    image = tf.expand_dims(image, axis = 0)#convert 3D tensor into 4D tensor
    return image

#function to generate the derained image
def predict(image):
    image = preprocess(image)
    generator = build_generator()
    derained_image = generator(image, training = False)
    derained_image = tf.squeeze(derained_image, axis = 0)#converting the 4D Tensor bck to a 3D tensor
    return derained_image.numpy()#returning a numpy array instead of a Tensor
