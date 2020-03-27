from os import path
import antspynet
import keras.backend as K
import keras
import numpy as np

from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
number_of_clusters = 10

number_of_training_data = len(y_train)
number_of_testing_data = len(y_test)

x = x_train / 255
y = y_train

pretrain_epochs = 300
pretrain_batch_size = 256

# Instantiate the DEC model

dcec_model = antspynet.DeepEmbeddedClusteringModel(
   number_of_units_per_layer=(32, 64, 128, 10),
   number_of_clusters = number_of_clusters,
   convolutional = True, input_image_size = (28, 28, 1))

model_weights_file = "dcecAutoencoderModelWeights.h5"
if not path.exists(model_weights_file):
    dcec_model.pretrain(x=x, optimizer='adam', epochs=pretrain_epochs,
      batch_size=pretrain_batch_size)
    dcec_model.autoencoder.save_weights(model_weights_file)  
else:
    dcec_model.autoencoder.load_weights(model_weights_file)  

dcec_model.compile(optimizer=keras.optimizers.adam(), loss=('kld', 'mse'), loss_weights=(0.1, 1))

y_predicted = dcec_model.fit(x, max_number_of_iterations=20000, batch_size=256,
  tolerance=0.0001, update_interval=10)


