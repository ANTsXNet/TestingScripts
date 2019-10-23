import numpy as np
import ants
import antspynet

import requests
import tempfile

from os import path

url = "https://github.com/mgoubran/HippMapp3r/blob/master/data/test_case/mprage.nii.gz?raw=true"

temp_directory = tempfile.TemporaryDirectory()
target_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", dir=temp_directory.name)
target_file.close()
image_file_name = target_file.name

print("Downloading test data set.")

if not path.exists(image_file_name):
    r = requests.get(url)
    with open(image_file_name, 'wb') as f:
        f.write(r.content)

image = ants.image_read(image_file_name)

#########################################
#
# Perform initial (stage 1) segmentation
#

print("*************  Initial stage segmentation  ***************")
print("  (warning:  steps are somewhat different in the ")
print("   publication.  just getting something to work)")
print("")

shape_initial_stage = (160, 160, 128)

print("    Initial step 1: bias correction.")
image_n4 = ants.n4_bias_field_correction(image)

# Threshold at 10th percentile of non-zero voxels in "robust range (fslmaths)"
print("    Initial step 2: threshold.")
image_n4_array = ((image_n4.numpy()).flatten())
image_n4_nonzero = image_n4_array[(image_n4_array > 0).nonzero()]
image_robust_range = np.quantile( image_n4_nonzero, (0.02, 0.98))
threshold_value = 0.10 * (image_robust_range[1] - image_robust_range[0]) + image_robust_range[0]
thresholded_mask = ants.threshold_image(image_n4, -10000, threshold_value, 0, 1)
thresholded_image = image_n4 * thresholded_mask

# Standardize image (should do patch-based stuff but making a quicker substitute for testing)
print("    Initial step 3: standardize.")
thresholded_array = ((thresholded_image.numpy()).flatten())
thresholded_nonzero = image_n4_array[(thresholded_array > 0).nonzero()]
image_mean = np.mean(thresholded_nonzero)
image_sd = np.std(thresholded_nonzero)
image_standard = (image_n4 - image_mean) / image_sd
image_standard = image_standard * thresholded_mask

# Resample image
print("    Initial step 4: resample to (160, 160, 128).")
image_resampled = ants.resample_image(image_standard, shape_initial_stage, True, 0)

# Build model and load weights for first pass
print("    Initial step 5: generate first network and download weights.")
model_initial_stage = antspynet.create_hippmapp3r_unet_model_3d((*shape_initial_stage, 1), True)
model_initial_stage.load_weights(antspynet.get_pretrained_network("hippMapp3rInitial"))

# Create initial segmentation image
print("    Initial step 6: prediction and write to disk.")
data_initial_stage = image_resampled.numpy()
data_initial_stage = np.expand_dims(data_initial_stage, 0)
data_initial_stage = np.expand_dims(data_initial_stage, -1)

prediction_initial_stage = np.squeeze(model_initial_stage.predict(data_initial_stage))
prediction_initial_stage[np.where(prediction_initial_stage >= 0.5)] = 1
prediction_initial_stage[np.where(prediction_initial_stage < 0.5)] = 0
mask_initial_stage = ants.from_numpy(prediction_initial_stage,
  origin=image_resampled.origin, spacing=image_resampled.spacing,
  direction=image_resampled.direction)
mask_initial_stage = ants.label_clusters(mask_initial_stage, min_cluster_size=10)
mask_initial_stage = ants.threshold_image(mask_initial_stage, 1, 2, 1, 0)
mask_initial_stage_original_space = ants.resample_image(mask_initial_stage, image_n4.shape, True, 1)
ants.image_write(mask_initial_stage_original_space, "mask_initial_stage.nii.gz")


#########################################
#
# Perform initial (stage 2) segmentation
#

print("")
print("")
print("*************  Refine stage segmentation  ***************")
print("  (warning:  These steps need closer inspection.)")
print("")

shape_refine_stage = (112, 112, 64)

# Trim image space
print("    Refine step 1: crop image centering on initial mask.")
# centroid = np.round(ants.label_image_centroids(mask_initial_stage)['vertices'][0]).astype(int)
centroid_indices = np.where(prediction_initial_stage == 1)
centroid = list()
centroid.append(int(np.mean(centroid_indices[0])))
centroid.append(int(np.mean(centroid_indices[1])))
centroid.append(int(np.mean(centroid_indices[2])))

lower = list()
lower.append(centroid[0] - int(0.5 * shape_refine_stage[0]))
lower.append(centroid[1] - int(0.5 * shape_refine_stage[1]))
lower.append(centroid[2] - int(0.5 * shape_refine_stage[2]))
upper = list()
upper.append(lower[0] + shape_refine_stage[0])
upper.append(lower[1] + shape_refine_stage[1])
upper.append(lower[2] + shape_refine_stage[2])

mask_trimmed = ants.crop_indices(mask_initial_stage, lower, upper)
image_trimmed = ants.crop_indices(image_resampled, lower, upper)

# Build model and load weights for second pass
print("    Refine step 2: generate second network and download weights.")
model_refine_stage = antspynet.create_hippmapp3r_unet_model_3d((*shape_refine_stage, 1), False)
model_refine_stage.load_weights(antspynet.get_pretrained_network("hippMapp3rRefine"))

# Create refine segmentation image
print("    Refine step 3: do monte carlo iterations (SpatialDropout).")
data_refine_stage = image_trimmed.numpy()
data_refine_stage = np.expand_dims(data_refine_stage, 0)
data_refine_stage = np.expand_dims(data_refine_stage, -1)

number_of_mc_iterations = 30

prediction_refine_stage = np.zeros((number_of_mc_iterations,*shape_refine_stage))
for i in range(number_of_mc_iterations):
    print("        Doing monte carlo iteration", i, "out of", number_of_mc_iterations)
    prediction_refine_stage[i,:,:,:] = np.squeeze(model_refine_stage.predict(data_refine_stage))

prediction_refine_stage = np.mean(prediction_refine_stage, axis=0)

print("    Refine step 4: Average monte carlo results and write probability mask image.")
prediction_refine_stage_array = np.zeros(image_resampled.shape)
prediction_refine_stage_array[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]] = prediction_refine_stage
probability_mask_refine_stage_resampled = ants.from_numpy(prediction_refine_stage_array,
  origin=image_resampled.origin, spacing=image_resampled.spacing,
  direction=image_resampled.direction)
probability_mask_refine_stage = ants.resample_image_to_target(
  probability_mask_refine_stage_resampled, image)
ants.image_write(probability_mask_refine_stage, "probability_refine_mask.nii.gz")
