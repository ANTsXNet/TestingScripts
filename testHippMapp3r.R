library( ANTsR )
library( ANTsRNet )

url <- "https://github.com/mgoubran/HippMapp3r/blob/master/data/test_case/mprage.nii.gz?raw=true"
imageFile <- "head.nii.gz"
download.file( url, imageFile )
image <- antsImageRead( imageFile )

#########################################
#
# Perform initial (stage 1) segmentation
#

cat( "*************  Initial stage segmentation  ***************\n" )
cat( "  (warning:  steps are somewhat different in the \n" )
cat( "   publication.  just getting something to work)\n" )
cat( "\n" )

shapeInitialStage <- c( 160, 160, 128 )

cat( "    Initial step 1: bias correction.\n" )
imageN4 <- n4BiasFieldCorrection( image )

# Threshold at 10th percentile of non-zero voxels in "robust range (fslmaths)"
cat( "    Initial step 2: threshold.\n" )
imageArray <- as.array( imageN4 )
imageRobustRange <- quantile( imageArray[which( imageArray != 0 )], probs = c( 0.02, 0.98 ) )
thresholdValue <- 0.10 * ( imageRobustRange[2] - imageRobustRange[1] ) + imageRobustRange[1]
thresholdedMask <- thresholdImage( imageN4, -10000, thresholdValue, 0, 1 )
thresholdedImage <- image * thresholdedMask

# Standardize image
cat( "    Initial step 3: standardize." )
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
cat( "    Initial step 4: resample to (160, 160, 128).\n" )
imageResampled <- resampleImage( imageNormalized, shapeInitialStage,
  useVoxels = TRUE, interpType = "linear" )

cat( "    Initial step 5: generate first network and download weights.\n" )
modelInitialStage <- createHippMapp3rUnetModel3D( c( shapeInitialStage, 1 ), doFirstNetwork = TRUE )
modelInitialStage$load_weights( getPretrainedNetwork( "hippMapp3rInitial" ) )

cat( "    Initial step 6: prediction and write to disk.\n" )
dataInitialStage <- array( data = as.array( imageResampled ), dim = c( 1, dim( imageResampled ), 1 ) )
maskArray <- modelInitialStage$predict( dataInitialStage )
maskImageResampled <- as.antsImage( maskArray[1,,,,1] ) %>% antsCopyImageInfo2( imageResampled )
maskImage <- resampleImage( maskImageResampled, dim( image ), useVoxels = TRUE,
  interpType = "nearestNeighbor" )
maskImage[maskImage >= 0.5] <- 1
maskImage[maskImage < 0.5] <- 0
antsImageWrite( maskImage, "maskInitialStage.nii.gz" )


#########################################
#
# Perform initial (stage 2) segmentation
#

cat( "\n" )
cat( "\n" )
cat( "*************  Refine stage segmentation  ***************\n" )
cat( "  (warning:  These steps need closer inspection.)\n" )
cat( "\n" )

shapeRefineStage <- c( 112, 112, 64 )

maskArray <- drop( maskArray )
centroidIndices <- which( maskArray == 1, arr.ind = TRUE, useNames = FALSE )
centroid <- rep( 0, 3 )
centroid[1] <- mean( centroidIndices[, 1] )
centroid[2] <- mean( centroidIndices[, 2] )
centroid[3] <- mean( centroidIndices[, 3] )
lower <- floor( centroid - 0.5 * shapeRefineStage )
upper <- lower + shapeRefineStage - 1

maskTrimmed <- cropIndices( maskImageResampled, lower, upper )
imageTrimmed <- cropIndices( imageResampled, lower, upper )

cat( "    Refine step 2: generate second network and download weights.\n" )
modelRefineStage <- createHippMapp3rUnetModel3D( c( shapeRefineStage, 1 ), FALSE )
modelRefineStage$load_weights( getPretrainedNetwork( "hippMapp3rRefine" ) )

dataRefineStage <- array( data = as.array( imageTrimmed ), dim = c( 1, shapeRefineStage, 1 ) )

cat( "    Refine step 3: do monte carlo iterations (SpatialDropout).\n" )
numberOfMCIterations <- 30
predictionRefineStage <- array( data = 0, dim = c( numberOfMCIterations, shapeRefineStage ) )
for( i in seq_len( numberOfMCIterations ) )
  {
  cat( "        Doing monte carlo iteration", i, "out of", numberOfMCIterations, "\n" )
  predictionRefineStage[i,,,] <- modelRefineStage$predict( dataRefineStage )[1,,,,1]
  }
predictionRefineStage <- apply( predictionRefineStage, c( 2, 3, 4 ), mean )

cat( "    Refine step 4: Average monte carlo results and write probability mask image.\n" )
predictionRefineStageArray <- array( data = 0, dim = dim( imageResampled ) )
predictionRefineStageArray[lower[1]:upper[1],lower[2]:upper[2],lower[3]:upper[3]] <- predictionRefineStage
probabilityMaskRefineStageResampled <- as.antsImage( predictionRefineStageArray ) %>% antsCopyImageInfo2( imageResampled )
probabilityMaskRefineStage <- resampleImageToTarget( probabilityMaskRefineStageResampled, image )
antsImageWrite( probabilityMaskRefineStage, "probabilityRefineMask.nii.gz" )