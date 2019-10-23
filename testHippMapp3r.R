library( ANTsR )
library( ANTsRNet )

url <- "https://github.com/mgoubran/HippMapp3r/blob/master/data/test_case/mprage.nii.gz?raw=true"
imageFile <- "head.nii.gz"
download.file( url, imageFile )
image <- antsImageRead( imageFile )

imageN4 <- n4BiasFieldCorrection( image )

# Threshold at 10th percentile of non-zero voxels in "robust range (fslmaths)"
imageArray <- as.array( imageN4 )
imageRobustRange <- quantile( imageArray[which( imageArray != 0 )], probs = c( 0.02, 0.98 ) )
thresholdValue <- 0.10 * ( imageRobustRange[2] - imageRobustRange[1] ) + imageRobustRange[1]
thresholdedMask <- thresholdImage( imageN4, -10000, thresholdValue, 0, 1 )
thresholdedImage <- image * thresholdedMask

# Standardize image
meanImage <- mean( thresholdedImage[thresholdedMask == 1] )
sdImage <- sd( thresholdedImage[thresholdedMask == 1] )
imageNormalized <- ( imageN4 - meanImage ) / sdImage
imageNormalized <- imageNormalized * thresholdedMask

antsImageWrite( imageNormalized, "imageNormalized.nii.gz" )

# patchSize <- round( dim( imageN4 ) / 2.2 )
# strides <- 25
# patches <- extractImagePatches( thresholdedImage, patchSize,
#   strideLength = strides, maskImage = thresholdedMask, returnAsArray = TRUE )
# patchesMean <- apply( patches, 1, mean )
# patchesSd <- apply( patches, 1, sd )
# patchesNormalized <- ( patches - patchesMean ) / patchesSd
# patchesNormalized[which( is.na( patchesNormalized ) )] <- 0
# imageNormalized <- reconstructImageFromPatches( patchesNormalized,
#   thresholdedMask, strideLength = strides, domainImageIsMask = TRUE )

# Resample image
imageResampled <- resampleImage( imageNormalized, c( 160, 160, 128 ),
  useVoxels = TRUE, interpType = "linear" )

modelInitialStage <- createHippMapp3rUnetModel3D( list( 160, 160, 128, 1 ), doFirstNetwork = TRUE )
modelInitialStage$load_weights( getPretrainedNetwork( "hippMapp3rInitial" ) )

dataInitialStage <- array( data = as.array( imageResampled ), dim = c( 1, dim( imageResampled ), 1 ) )
maskArray <- modelInitialStage$predict( dataInitialStage )
maskImageResampled <- as.antsImage( maskArray[1,,,,1] ) %>% antsCopyImageInfo2( imageResampled )
maskImage <- resampleImage( maskImageResampled, dim( image ), useVoxels = TRUE,
  interpType = "nearestNeighbor" )
maskImage[maskImage >= 0.5] <- 1
maskImage[maskImage < 0.5] <- 0
antsImageWrite( maskImage, "maskInitialStage.nii.gz" )



