import os
import tensorflow as tf
import numpy as np

def resBlock(model):
    orig = model
    model = tf.keras.layers.Conv2D(64, kernel_size = 3,padding = 'same')(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.PReLU(shared_axes=[1,2])(model)
    model = tf.keras.layers.Conv2D(64, kernel_size = 3,padding = 'same')(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Add()([orig, model])
    return model

def discBlock(model, kernels, stride):
    model = tf.keras.layers.Conv2D(kernels, kernel_size = 3, strides = stride,padding='same')(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU()(model)
    return model

def Generator():
    lr_input = tf.keras.layers.Input(shape = (None,None,3))
    model = tf.keras.layers.Conv2D(64,9,padding = 'same')(lr_input)
    model = tf.keras.layers.PReLU(shared_axes=[1,2])(model)
    start_layer = model
    for i in range(16):
      model = resBlock(model)
    model = tf.keras.layers.Conv2D(64,3,padding = 'same')(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Add()([start_layer,model])
    model = tf.keras.layers.Conv2D(256,3,padding = 'same')(model)
    model = tf.keras.layers.UpSampling2D(size = 2)(model)
    model = tf.keras.layers.PReLU(shared_axes=[1,2])(model)
    model = tf.keras.layers.Conv2D(256,3,padding = 'same')(model)
    model = tf.keras.layers.UpSampling2D(size = 2)(model)
    model = tf.keras.layers.PReLU(shared_axes=[1,2])(model)
    output = tf.keras.layers.Conv2D(3,3,padding = 'same')(model)
    gen_model = tf.keras.models.Model(inputs = lr_input, outputs = output)
    return gen_model

def normalize(image):
    image = (image / 127.5) - 1
    return image

def preprocess(image):
    image = np.array(image)
    image = normalize(image)
    image = tf.expand_dims(image, axis = 0)
    return image

def build_generator():
    generator = Generator()
    CURRENT_WORKING_DIRECTORY = str(os.getcwd())
    generator.load_weights(CURRENT_WORKING_DIRECTORY + '/models/srganGenerator.h5')
    return generator

def predict(image):
    image = preprocess(image)
    generator = build_generator()
    superres_image = generator.predict(image)
    superres_image = tf.squeeze(superres_image, axis = 0)
    return superres_image.numpy()
