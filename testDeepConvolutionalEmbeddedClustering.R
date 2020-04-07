library( keras )
library( ANTsRNet )

keras::backend()$clear_session()

# Let's use the fmnist data set.

fmnist <- dataset_fashion_mnist()

numberOfTrainingData <- length( fmnist$train$y )
numberOfTestingData <- length( fmnist$test$y )

inputImageSize <- dim( fmnist$test$x[1,,] )

x <- array( data = fmnist$train$x / 255, dim = c( numberOfTrainingData, inputImageSize, 1 ) )
y <- fmnist$train$y

numberOfClusters <- length( unique( fmnist$train$y ) )

pretrainEpochs <- 300L
pretrainBatchSize <- 256L

pretrainOptimizer <- optimizer_sgd( lr = 1.0, momentum = 0.9 )

# Instantiate the DCEC model

dcecModel <- DeepEmbeddedClusteringModel$new(
   numberOfUnitsPerLayer = c( 32, 64, 128, 10 ),
   numberOfClusters = numberOfClusters,
   convolutional = TRUE, inputImageSize = c( inputImageSize, 1 ) )

modelWeightsFile <- "dcecAutoencoderModelWeights.h5"
if( ! file.exists( modelWeightsFile ) )
  {
  dcecModel$pretrain( x = x, optimizer = 'adam',
    epochs = pretrainEpochs, batchSize = pretrainBatchSize )
  save_model_weights_hdf5( dcecModel$autoencoder, modelWeightsFile )
  } else {
  load_model_weights_hdf5( dcecModel$autoencoder, modelWeightsFile )
  }

dcecModel$compile( optimizer = 'adam', loss = c( 'kld', 'mse' ), loss_weights = c( 0.1, 1 ) )

yPredicted <- dcecModel$fit( x, maxNumberOfIterations = 2e4, batchSize = 256,
  tolerance = 1e-3, updateInterval = 10 )

