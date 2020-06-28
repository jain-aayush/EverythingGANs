import tensorflow as tf
import tensorflow_addons as tfad

def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
      result.add(tfad.layers.InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfad.layers.InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

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

def build_generator():
    generator = Generator()
    generator.load_weights('ImageSuperResolution.h5')
    return generator

def normalize(image):
    image = (image/127.5) - 1
    return image

def preprocess(image):
    image = normalize(image)
    return image

def predict(image):
    image = preprocess(image)
    generator = build_generator()
    animated_image = generator(image)
    return image
