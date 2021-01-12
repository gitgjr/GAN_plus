import tensorflow as tf
import tensorflow.keras as keras
import CNN

a = CNN.CNN()

a.network.load_weights('./CNN/ckpt/cp-0005.ckpt')
def classify(input_img):
    a.network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    classification = a.network.predict(input_img)
    # print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
    return classification