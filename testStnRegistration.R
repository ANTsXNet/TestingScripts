library( keras )
library( ANTsRNet )
library( reticulate )

K <- keras::backend()

K$clear_session()

Sys.setenv( "CUDA_VISIBLE_DEVICES" = 3 )

getInitialWeights2D <- function(outputSize) {
    np <- reticulate::import("numpy")
    b <- np$zeros(c(2L, 3L), dtype = "float32")
    b[1, 1] <- 1
    b[2, 2] <- 1
    W <- np$zeros(c(as.integer(outputSize), 6L), dtype = "float32")
    weights <- list()
    weights[[1]] <- W
    weights[[2]] <- as.array(as.vector(t(b)))
    return(weights)
}

layer_spatial_transformer_2d <- function( objects, resampledSize, name = NULL ) {
create_layer( SpatialTransformerLayer2D, objects,
    list( resampledSize = resampledSize, name = name )
    )
}

img = ri( 1 ) %>% resampleImage( 2 )
img2 = ri( 5 ) %>% resampleImage( 8 )
resampledSize <- dim( img2 )
inputImageSize = c( dim( img ),  1 )
inputs <- layer_input(shape = inputImageSize)
localization <- inputs
localization <- localization %>% layer_max_pooling_2d(pool_size = c(2,
    2))
localization <- localization %>% layer_conv_2d(filters = 20,
    kernel_size = c(5, 5))
localization <- localization %>% layer_max_pooling_2d(pool_size = c(2,
    2))
localization <- localization %>% layer_conv_2d(filters = 20,
    kernel_size = c(5, 5))
localization <- localization %>% layer_flatten()
localization <- localization %>% layer_dense(units = 50L)
localization <- localization %>% layer_activation_relu()
weights <- getInitialWeights2D(outputSize = 50L)
localization <- localization %>% layer_dense(units = 6L,
    weights = weights)
outputs <-  layer_spatial_transformer_2d(list(inputs, localization),
        resampledSize, name = "layer_spatial_transformer")

stnModel <- keras_model(inputs = inputs, outputs = outputs)

X = array( as.array( img ), dim = inputImageSize )
Y = array( as.array( img2 ), dim = c( resampledSize, 1 ) )

stnModel %>% compile(
  loss = 'mse',
  optimizer = optimizer_rmsprop()
)

history <- stnModel %>% fit(
  X, Y,  epochs = 30, batch_size = 1 )

# after this we get out the learned weights from the stnModel and compare to:
trueReg = antsRegistration( img, img2, "Translation" )
tx = readAntsrTransform( trueReg$fwdtransforms )
params =  getAntsrTransformParameters( tx )

# what are the learned parameters?
