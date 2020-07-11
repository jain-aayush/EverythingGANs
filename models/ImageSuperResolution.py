#importing the required libraries
import os
import tensorflow as tf
import numpy as np

#residual block of the architecture
def resBlock(model):
    orig = model
    model = tf.keras.layers.Conv2D(64, kernel_size = 3,padding = 'same')(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.PReLU(shared_axes=[1,2])(model)
    model = tf.keras.layers.Conv2D(64, kernel_size = 3,padding = 'same')(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Add()([orig, model])
    return model

#discriminator block of the architecture
def discBlock(model, kernels, stride):
    model = tf.keras.layers.Conv2D(kernels, kernel_size = 3, strides = stride,padding='same')(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.LeakyReLU()(model)
    return model

#SRGAN generator
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

#normalizing image pixels to [0,1]
def normalize(image):
    image = image / 255.0
    return image

#preprocessing function
def preprocess(image):
    image = np.array(image)
    image = normalize(image)
    image = tf.expand_dims(image, axis = 0)#convert 3D tensor into 4D tensor
    return image

#helper function to load weights and build the generator
def build_generator():
    generator = Generator()
    CURRENT_WORKING_DIRECTORY = str(os.getcwd())
    generator.load_weights(CURRENT_WORKING_DIRECTORY + '/models/srganGenerator.h5')
    return generator

#function to generate the superreolved image
def predict(image):
    image = preprocess(image)
    generator = build_generator()
    superres_image = generator.predict(image)
    superres_image = tf.squeeze(superres_image, axis = 0)#converting the 4D Tensor bck to a 3D tensor
    return superres_image.numpy()#returning a numpy array instead of a Tensor
