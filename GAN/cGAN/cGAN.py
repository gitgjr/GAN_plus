import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import os

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape([-1,28,28,1]) / 255

# hyp
epoch = 2000
batch_size = 64
retrain = 1
save_epoch = 100

g_sequential = Sequential([
    Dense(7 * 7 * 64, input_shape=[100 + 10]),
    BatchNormalization(),
    LeakyReLU(),
    Reshape([7, 7, 64]),
    UpSampling2D([2, 2]),
    Conv2DTranspose(64, [3, 3], padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    UpSampling2D([2, 2]),
    Conv2DTranspose(1, [3, 3], padding='same', activation='sigmoid')
])

### discriminator
discriminator = Sequential([
    Conv2D(64, [3, 3], padding='same', input_shape=[28, 28, 1]),
    BatchNormalization(),
    LeakyReLU(),
    MaxPool2D([2, 2]),
    Conv2D(64, [3, 3], padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    MaxPool2D([2, 2]),
    Flatten(),
    Dense(128),
    BatchNormalization(),
    LeakyReLU(),
    Dense(11, activation='softmax')
])

g_sample_input = Input([100]) # generator input
g_label_input = Input([], dtype='int32') # Specify label input
x_input = Input([28, 28, 1]) # true sample input
x_label_input = Input([], dtype='int32') # true lable input

condition_g_sample_input = K.concatenate([g_sample_input, K.one_hot(g_label_input, 10)]) # Combining random data input with specified label hot codes

g_output = g_sequential(condition_g_sample_input) # generator input
generator = Model(inputs=[g_sample_input, g_label_input], outputs=g_output) # generator model

log_clip = Lambda(lambda x: K.log(K.clip(K.stop_gradient(x), 1e-3, 1) - K.stop_gradient(x) + x))

g_prob = discriminator(generator([g_sample_input, g_label_input])) # Output of discriminator identifying false samples
g_index = K.stack([K.arange(0, K.shape(g_prob)[0]), g_label_input], axis=1) # Index g_prob specifies the tag probability value

d_prob = discriminator(x_input) # 判别器识别真实样本的输出
x_index = K.stack([K.arange(0, K.shape(d_prob)[0]), x_label_input], axis=1) # Index d_prob specifies the tag probability value


d_loss = (
    - log_clip(tf.gather_nd(d_prob, x_index))
    - log_clip(1.0 - tf.gather_nd(g_prob, g_index))
)

fit_discriminator = Model(inputs=[g_sample_input, g_label_input, x_input, x_label_input], outputs=d_loss)
fit_discriminator.add_loss(d_loss) # Add custom loss
generator.trainable = False
for layer in generator.layers:
    if isinstance(layer, BatchNormalization): # Set Batchnormalization to training mode
        layer.trainable = True
fit_discriminator.compile(optimizer=Adam(0.001))
generator.trainable = True


g_loss = (
    -log_clip(tf.gather_nd(g_prob, g_index)) # Log (probability value of label specified by false sample)
)


fit_generator = Model(inputs=[g_sample_input, g_label_input], outputs=g_loss)
fit_generator.add_loss(g_loss)

### Generator training does not update discriminator parameters
discriminator.trainable = False
for layer in discriminator.layers:
    if isinstance(layer, BatchNormalization): # Set Batchnormalization to training mode
        layer.trainable = True
fit_generator.compile(optimizer=Adam(0.001))
discriminator.trainable = True

# check point
checkpoint_path_g = "checkpoint/cpg.ckpt"
checkpoint_path_d = "checkpoint/cpd.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path_g)
cp_callback_g = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_g,
                                                 save_weights_only=True,
                                                 verbose=0,
                                                    save_freq=save_epoch)
cp_callback_d = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_d,
                                                 save_weights_only=True,
                                                 verbose=0,
                                                   save_freq=save_epoch)
if (retrain == 0):
    fit_generator.load_weights(checkpoint_path_g)
    fit_discriminator.load_weights(checkpoint_path_d)
    print('load checkpoint successfully')

for i in range(epoch):
    if i % 10 == 0:
        plt.imshow(generator.predict([K.constant(np.random.uniform(-1, 1, [1, 100])), K.constant([i % 10])])[0].reshape(
            [28, 28]), cmap='gray')
        plt.title(str(i % 10))
        plt.show()
    print(i)
    index = random.sample(range(len(train_x)), batch_size)
    x_label = train_y[index]
    x = train_x[index]
    g_sample = np.random.uniform(-1, 1, [batch_size, 100])
    g_label = np.random.randint(0, 10, [batch_size])

    fit_discriminator.fit([K.constant(g_sample), K.constant(g_label), K.constant(x), K.constant(x_label)],callbacks=[cp_callback_d])
    fit_generator.fit([K.constant(g_sample), K.constant(g_label)],callbacks=[cp_callback_g])

# show example
img_example = generator.predict([K.constant(np.random.uniform(-1, 1, [1, 100])), K.constant([8])])
result_example = discriminator.predict(img_example)
print(img_example.shape)
plt.imshow(img_example[0], cmap='gray')
plt.title(result_example, fontsize=10)

fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axes[i, j].imshow(generator.predict([K.constant(np.random.uniform(-1, 1, [1, 100])), K.constant([i])])[0].reshape(
            [28, 28]), cmap='gray')
        axes[i, j].axis(False)

plt.show()
