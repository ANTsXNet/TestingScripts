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
number_of_pixels = x_train.shape[1] * x_train.shape[2]

x_train_reshaped = x_train.reshape((number_of_training_data, number_of_pixels))
x_test_reshaped = x_test.reshape((number_of_testing_data, number_of_pixels))

x = np.concatenate((x_train_reshaped, x_test_reshaped), axis=0) / 255
y = np.concatenate((y_train, y_test), axis=0)

pretrain_epochs = 300
pretrain_batch_size = 256

# Instantiate the DEC model

dec_model = antspynet.DeepEmbeddedClusteringModel(
   number_of_units_per_layer=(number_of_pixels, 500, 500, 2000, 10),
   number_of_clusters = number_of_clusters)

model_weights_file = "decAutoencoderModelWeights.h5"
if not path.exists(model_weights_file):
    dec_model.pretrain(x=x, optimizer='adam', epochs=pretrain_epochs,
      batch_size=pretrain_batch_size)
    dec_model.autoencoder.save_weights(model_weights_file)  
else:
    dec_model.autoencoder.load_weights(model_weights_file)  

dec_model.compile(optimizer=keras.optimizers.sgd(lr=1.0, momentum=0.9), loss='kld')

y_predicted = dec_model.fit(x, max_number_of_iterations=20000, batch_size=256,
  tolerance=0.001, update_interval=10)


