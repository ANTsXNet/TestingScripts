import requests
import tempfile
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from os import path
import cv2

import ants
import antspynet


####
#
#  Test 2-D
#


# url = "https://user-images.githubusercontent.com/22609465/36634042-4168652a-1964-11e8-90a9-2c480b97eff7.jpg"
# r = requests.get(url)

# temp_directory = tempfile.TemporaryDirectory()
# target_file = tempfile.NamedTemporaryFile(suffix=".jpg", dir=temp_directory.name)
# target_file.close()
# target_file_name = target_file.name

# if not path.exists(target_file_name):
#     r = requests.get(url)
#     with open(target_file_name, 'wb') as f:
#         f.write(r.content)

# f = cv2.imread(target_file_name)

# ft = Input(shape=(None, None, 3))
# # ft = Input(shape=f.shape)

# shape = (700, 700)

# types = ['nearest_neighbor', 'linear', 'cubic']
# for i in range(len(types)):
#     output = antspynet.ResampleTensorLayer2D(shape, types[i])(ft)
#     model = Model(inputs=ft, outputs=output)
#     batchX = np.expand_dims(f, axis = 0)
#     output_image = model.predict(batchX)
#     cv2.imwrite("out2D_" + types[i] + ".jpg", output_image[0])


####
#
# Test 3-D
#

brain_file = ants.get_ants_data("ch2")

f = ants.image_read(brain_file)
f = np.expand_dims( f.numpy(), axis=-1)

shape = (200, 200, 200)
ft = Input(shape=(None, None, None, 1))
# ft = Input(shape=f.shape)

types = ['nearest_neighbor', 'linear', 'cubic']
for i in range(len(types)):
    output = antspynet.ResampleTensorLayer3D(shape, types[i])(ft)
    model = Model(inputs=ft, outputs=output)
    batchX = np.expand_dims(f, axis = 0)
    output_image = ants.from_numpy(np.squeeze(model.predict(batchX)))
    ants.image_write(output_image, "out3D_py_" + types[i] + ".nii.gz")


