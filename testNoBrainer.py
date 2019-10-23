import requests
import tempfile
from os import path
import ants
import antspynet
import numpy as np

print("Generate network")
model = antspynet.create_nobrainer_unet_model_3d((None, None, None, 1))
model.summary()

print("Download: model weights")
url_weights = "https://github.com/neuronets/nobrainer-models/releases/download/0.1/brain-extraction-unet-128iso-weights.h5"
temp_directory = tempfile.TemporaryDirectory()
target_file_weights = tempfile.NamedTemporaryFile(suffix=".h5", dir=temp_directory.name)
target_file_weights.close()
target_file_weights_name = target_file_weights.name
if not path.exists(target_file_weights_name):
    r = requests.get(url_weights)
    with open(target_file_weights_name, 'wb') as f:
        f.write(r.content)

model.load_weights(target_file_weights_name)

print("Download: test image")
url_image = "https://github.com/ANTsXNet/BrainExtraction/blob/master/Data/Example/1097782_defaced_MPRAGE.nii.gz?raw=true"
target_file_image = tempfile.NamedTemporaryFile(suffix=".nii.gz", dir=temp_directory.name)
target_file_image.close()
target_file_image_name = target_file_image.name
if not path.exists(target_file_image_name):
    r = requests.get(url_image)
    with open(target_file_image_name, 'wb') as f:
        f.write(r.content)

image = ants.image_read(target_file_image_name)

print("Preprocessing: bias correction")
image_n4 = ants.n4_bias_field_correction(image)
image_n4 = ants.image_math(image_n4, 'Normalize') * 255.0

print("Preprocessing:  thresholding")
image_n4_array = ((image_n4.numpy()).flatten())
image_n4_nonzero = image_n4_array[(image_n4_array > 0).nonzero()]
image_robust_range = np.quantile( image_n4_nonzero, (0.02, 0.98))
threshold_value = 0.10 * (image_robust_range[1] - image_robust_range[0]) + image_robust_range[0]
thresholded_mask = ants.threshold_image(image_n4, -10000, threshold_value, 0, 1)
thresholded_image = image_n4 * thresholded_mask

print("Preprocessing:  resampling")
image_resampled = ants.resample_image(thresholded_image, (256, 256, 256), True)
batchX = np.expand_dims(image_resampled.numpy(), axis=0)
batchX = np.expand_dims(batchX, axis=-1)

brain_mask_array = model.predict(batchX, verbose=0)
brain_mask_resampled = ants.from_numpy(np.squeeze(brain_mask_array[0,:,:,:,0]),
          origin=image_resampled.origin, spacing=image_resampled.spacing,
          direction=image_resampled.direction)
brain_mask_image = ants.resample_image(brain_mask_resampled, image.shape, True, 1)
minimum_brain_volume = round( 649933.7 )
brain_mask_labeled = ants.label_clusters(brain_mask_image, minimum_brain_volume)
ants.image_write(brain_mask_labeled, "brain_mask.nii.gz")
