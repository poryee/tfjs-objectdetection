import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
print(tf.version.VERSION)
import tensorflow_hub as hub
'''
model = tf.keras.Sequential(
    [tf.keras.Input(3), tf.keras.layers.Dense(3), tf.keras.layers.Dense(1)]
)
model.compile(loss="mse", optimizer="adam")
model.fit(tf.constant([[1, 2, 3], [4, 5, 6]]), tf.constant([1, 2]))
model.save("model.h5")
restored_model = tf.keras.models.load_model("model.h5")
restored_model.summary()
'''
newmodel = tf.keras.models.load_model(r'C:\Users\PorYee\Desktop\testout\efficientdet h5 model')


#detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")
print('model loaded!')
newmodel.summary()
print("MobileNet has {} trainable variables: {}, ...".format(
          len(newmodel.trainable_variables),
          ", ".join([v.name for v in newmodel.trainable_variables[:5]])))
