library( ANTsR )
library( ANTsRNet )
library( keras )
library( magick )
library( tensorflow )

bikeFile <- tempfile( pattern = "bike", fileext = ".jpg" )
download.file(
  url = "https://user-images.githubusercontent.com/22609465/36634042-4168652a-1964-11e8-90a9-2c480b97eff7.jpg",
  destfile = bikeFile )

f <- antsImageRead( bikeFile )
f <- as.array( f )
f <- aperm( f, c( 3, 2, 1 ) )
f <- array( data = f, dim = c( 1, dim( f ) ) )

ft <- layer_input( batch_shape = dim( f ) )

shape = c( 700, 700 )

types <- c( 'nearestNeighbor', 'linear', 'cubic' )
for( i in seq_len( length( types ) ) )
  {
  output <- layer_resample_tensor_2d( ft, shape, types[i] )
  model <- keras_model( inputs = ft, outputs = output )
  outputImage <- drop( model %>% predict( f ) )

  # Rescale from [0, 1]
  ys <- ( outputImage - min( outputImage ) ) /
        ( max( outputImage ) - min( outputImage ) )
  img <- magick::image_read( ys )
  image_write( img, path =
    paste0( "out_", types[i], format = ".jpg" ) )
  }

