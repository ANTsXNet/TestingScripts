library( keras )
library( ANTsRNet )
library( magick )
library( ANTsR )

keras::backend()$clear_session()

dataset <- 'vangogh2photo'

dataDirectory <- paste0( getwd(), "/", dataset )
if( ! dir.exists( dataDirectory ) )
  {
  zippedFile <- tempfile( pattern = dataset, fileext = ".zip" )

  if( ! file.exists( zippedFile ) )
    {
    download.file(
      url = paste0( "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/", dataset, ".zip" ),
      destfile = zippedFile )
    unzip( zippedFile )
    }
  }


scaleFactor <- 4
highResolutionImageSize <- c( 224, 224, 3 )
lowResolutionImageSize <- c( as.integer( highResolutionImageSize[1] / scaleFactor ),
                             as.integer( highResolutionImageSize[2] / scaleFactor ),
                             3 )

ganModel <- SuperResolutionGanModel$new(
   lowResolutionImageSize = lowResolutionImageSize,
   scaleFactor = scaleFactor )

cat( "Reading training set A files.\n" )
trainingFilesA <- list.files( paste0( dataDirectory, '/trainA/' ),
  pattern = ".jpg", full.names = TRUE )
numberOfTrainingAFiles <- length( trainingFilesA )

X_trainLowResolution <- array( data = 0, dim = c( numberOfTrainingAFiles, lowResolutionImageSize ) )
X_trainHighResolution <- array( data = 0, dim = c( numberOfTrainingAFiles, highResolutionImageSize ) )

pb <- txtProgressBar( min = 0, max = length( trainingFilesA ), style = 3 )
for( i in seq_len( numberOfTrainingAFiles ) )
  {
  # high resolution
  image <- image_read( trainingFilesA[i] )
  image <- image_data( image_scale( image, "224x224" ) )
  image <- as.double( aperm( image, c( 3, 2, 1 ) ) )
  image <- image / 127.5 - 1.0
  X_trainHighResolution[i,,,] <- as.double( image )

  # low resolution
  image <- image_read( trainingFilesA[i] )
  image <- image_data( image_scale( image, "56x56" ) )
  image <- as.double( aperm( image, c( 3, 2, 1 ) ) )
  image <- image / 127.5 - 1.0
  X_trainLowResolution[i,,,] <- as.double( image )

  setTxtProgressBar( pb, i )
  }
cat( "\nDone.\n\n" )


ganModel$train( X_trainLowResolution, X_trainHighResolution,
  numberOfEpochs = 200, batchSize = 4, sampleInterval = 1,
  sampleFilePrefix = "./SuperResolutionGanSampleImages/sample" )

