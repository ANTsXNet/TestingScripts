library( ANTsR )
library( ANTsRNet )
library( keras )
library( magick )
library( tensorflow )

butterflyFile <- tempfile( pattern = "butterfly", fileext = ".png" )
download.file(
  url = "https://user-images.githubusercontent.com/22609465/36634043-4178580e-1964-11e8-9ebf-69c4b6ad52a5.png",
  destfile = butterflyFile )

b <- antsImageRead( butterflyFile )
bChannels <- splitChannels( b )
bChannels[[1]] <- resampleImage( bChannels[[1]], c( 128, 128 ), useVoxel = TRUE, interpType = 0 )
bChannels[[2]] <- resampleImage( bChannels[[2]], c( 128, 128 ), useVoxel = TRUE, interpType = 0 )
bChannels[[3]] <- resampleImage( bChannels[[3]], c( 128, 128 ), useVoxel = TRUE, interpType = 0 )
b <- mergeChannels( bChannels )
b <- as.array( b )
b <- aperm( b, c( 3, 2, 1 ) )
b <- array( data = b, dim = c( 1, dim( b ) ) )

bikeFile <- tempfile( pattern = "bike", fileext = ".jpg" )
download.file(
  url = "https://user-images.githubusercontent.com/22609465/36634042-4168652a-1964-11e8-90a9-2c480b97eff7.jpg",
  destfile = bikeFile )

f <- antsImageRead( bikeFile )
f <- as.array( f )
f <- aperm( f, c( 3, 2, 1 ) )
f <- array( data = f, dim = c( 1, dim( f ) ) )

ft <- layer_input( batch_shape = dim( f ) )
bt <- layer_input( batch_shape = dim( b ) )

output <- layer_contextual_attention_2d( list( ft, bt ), kernelSize = 3L,
           stride = 1L, dilationRate = 2L, fusionKernelSize = 3L )

model <- keras_model( inputs = list( ft, bt ), outputs = output )

outputImage <- model %>% predict( list( f, b ), batch_size = 1 )
outputImage <- drop( outputImage )

ys <- ( outputImage - min( outputImage ) ) / ( max( outputImage ) - min( outputImage ) )
img <- magick::image_read( ys )
image_write(img, path = "out.jpg", format = "jpg")