import tensorflow as tf
import h5py


model = tf.keras.models.load_model('tryModel.h5')
print(model.summary())