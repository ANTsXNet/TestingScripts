library( ANTsR )
library( ANTsRNet )

cat( "Generate network\n")
model <- createNoBrainerUnetModel3D( list( NULL, NULL, NULL, 1 ) )

cat( "Download:  model weights\n")
url <- "https://github.com/neuronets/nobrainer-models/releases/download/0.1/brain-extraction-unet-128iso-weights.h5"
weightsFile <- "nobrainerWeights.h5"
download.file( url, weightsFile )
model$load_weights( weightsFile )

cat( "Download:  test image\n")
url <- "https://github.com/ANTsXNet/BrainExtraction/blob/master/Data/Example/1097782_defaced_MPRAGE.nii.gz?raw=true"
imageFile <- "head.nii.gz"
download.file( url, imageFile )

image <- antsImageRead( imageFile )

cat( "Preprocessing:   bias correction\n")
imageN4 <- ( n4BiasFieldCorrection( image ) %>% iMath( "Normalize" ) ) * 255.0

cat( "Preprocessing:   thresholding\n")
imageArray <- as.array( imageN4 )
imageRobustRange <- quantile( imageArray[which( imageArray != 0 )], probs = c( 0.02, 0.98 ) )
thresholdValue <- 0.10 * ( imageRobustRange[2] - imageRobustRange[1] ) + imageRobustRange[1]
thresholdedMask <- thresholdImage( imageN4, -10000, thresholdValue, 0, 1 )
thresholdedImage <- imageN4 * thresholdedMask

cat( "Preprocessing:   resampling\n")
imageResampled <- resampleImage( image, rep( 256, 3 ), useVoxels = TRUE )
imageArray <- array( as.array( imageResampled ), dim = c( 1, dim( imageResampled ), 1 ) )

cat( "Prediction and write to disk.\n")
brainMaskArray <- predict( model, imageArray )
brainMaskResampled <- as.antsImage( brainMaskArray[1,,,,1] ) %>% antsCopyImageInfo2( imageResampled )
brainMaskImage = resampleImage( brainMaskResampled, dim( image ),
  useVoxels = TRUE, interpType = "nearestneighbor" )
minimumBrainVolume <- round( 649933.7 / prod( antsGetSpacing( image ) ) )
brainMaskLabeled = labelClusters( brainMaskImage, minimumBrainVolume )
antsImageWrite( brainMaskLabeled, 'brainMask.nii.gz' )



