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
train_x = train_x.reshape([-1, 28 * 28]) / 255

# hyp
batch_size = 64
epoch = 2000
retrain = 1
save_epoch = 100

### generator
generator = Sequential([
    Dense(128, activation='relu', input_shape=[100]),
    Dense(28 * 28, activation='sigmoid')
])

### discriminator
discriminator = Sequential([
    Dense(128, activation='relu', input_shape=[28 * 28]),
    Dense(11, activation='sigmoid')
])

g_sample_input = Input([100]) # generator input
x_input = Input([28 * 28]) # true data input
label_input = Input([], dtype='int32') # true lable input

### Cut the probability to the interval [1e-3, 1], and calculate its log to avoid inf,
# k.stop after log_Gradient means that the gradient is not calculated during training
#   log_clip = Lambda(lambda x: K.log(x + 1e-3)) is ok too
log_clip = Lambda(lambda x: K.log(K.clip(K.stop_gradient(x), 1e-3, 1) - K.stop_gradient(x) + x))

g_prob = discriminator(generator(g_sample_input)) # Output of discriminator identifying false samples
d_prob = discriminator(x_input) # Output of discriminator identifying true samples
index = K.stack([K.arange(0, K.shape(d_prob)[0]), label_input], axis=1) #  index d_Prob probability of correct label

### d loss
d_loss = (
    - log_clip(1.0 - d_prob[:, -1])
    - log_clip(g_prob[:, -1])
    - log_clip(tf.gather_nd(d_prob, index)) # Logarithm of correct label probability of real samples
)

fit_discriminator = Model(inputs=[g_sample_input, x_input, label_input], outputs=d_loss)
fit_discriminator.add_loss(d_loss) # Add custom loss

### Set generator.trainable as False before calling complie
# The parameters of generator are not updated during model training after calling compile
generator.trainable = False
fit_discriminator.compile(optimizer=Adam(0.001))
generator.trainable = True

### g loss
g_loss = (
    log_clip(g_prob[:, -1])
)

fit_generator = Model(inputs=g_sample_input, outputs=g_loss) # Model used to train discriminator
fit_generator.add_loss(g_loss) # Add custom loss

### Generator training does not update discriminator parameters
discriminator.trainable = False
fit_generator.compile(optimizer=Adam(0.001))
discriminator.trainable = True

# checkpoint
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
    if i % 100 == 0:
        plt.imshow(generator.predict(np.random.uniform(-1, 1, [1, 100]))[0].reshape([28, 28]), cmap='gray')
        plt.show()
    print(i)
    index = random.sample(range(len(train_x)), batch_size)
    label = train_y[index]
    x = train_x[index]
    g_sample = np.random.uniform(-1, 1, [batch_size, 100])
    fit_discriminator.fit([K.constant(g_sample), K.constant(x), K.constant(label)],callbacks=[cp_callback_d])
    fit_generator.fit(g_sample,callbacks=[cp_callback_g])
# show example
img_example = generator.predict(np.random.uniform(-1, 1, [1, 100]))
result_example = discriminator.predict(img_example)
print(img_example.shape)
plt.imshow(img_example[0],cmap='gray')
plt.title(result_example,fontsize=10)

fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axes[i, j].imshow(generator.predict(np.random.uniform(-1, 1, [1, 100]))[0].reshape([28, 28]), cmap='gray')
        axes[i, j].axis(False)
plt.show()
