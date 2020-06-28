import tensorflow as tf

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
    image = image/255.0
    return image

def preprocess(image):
    image = normalize(image)
    return image

def build_generator():
    generator = Generator()
    generator.load_weights('srganGenerator.h5')
    return generator

def predict(image):
    image = preprocess(image)
    generator = build_generator()
    superres_image = generator.predict(image)
    return image
