library( keras )
library( ANTsRNet )

keras::backend()$clear_session()

Sys.setenv( "CUDA_VISIBLE_DEVICES" = 3 )

# Let's use the fmnist data set.

fmnist <- dataset_fashion_mnist()

numberOfTrainingData <- length( fmnist$train$y )
numberOfTestingData <- length( fmnist$test$y )

numberOfPixels <- prod( dim( fmnist$test$x[1,,] ) )

fmnist$train$xreshaped <- array_reshape( fmnist$train$x,
  dim = c( numberOfTrainingData, numberOfPixels ), order = "C" )
fmnist$test$xreshaped <- array_reshape( fmnist$train$x,
  dim = c( numberOfTrainingData, numberOfPixels ), order = "C" )

x <- rbind( fmnist$test$xreshaped, fmnist$train$xreshaped ) / 255
y <- c( fmnist$test$y, fmnist$train$y )

numberOfClusters <- length( unique( fmnist$train$y ) )

pretrainEpochs <- 300L
pretrainBatchSize <- 256L

pretrainOptimizer <- 'adam'

# Instantiate the DEC model

decModel <- DeepEmbeddedClusteringModel$new(
   numberOfUnitsPerLayer = c( numberOfPixels, 500, 500, 2000, 10 ),
   numberOfClusters = numberOfClusters )

modelWeightsFile <- "decAutoencoderModelWeights.h5"
if( ! file.exists( modelWeightsFile ) )
  {
  decModel$pretrain( x = x, optimizer = 'adam',
    epochs = pretrainEpochs, batchSize = pretrainBatchSize )
  save_model_weights_hdf5( decModel$autoencoder, modelWeightsFile )
  } else {
  load_model_weights_hdf5( decModel$autoencoder, modelWeightsFile )
  }

decModel$compile( optimizer = optimizer_sgd( lr = 1.0, momentum = 0.9 ), loss = 'kld' )

yPredicted <- decModel$fit( x, maxNumberOfIterations = 2e4, batchSize = 256,
  tolerance = 1e-3, updateInterval = 10 )

