import tensorflow as tf

def mean_squared_error(a, b):
  return tf.reduce_mean(tf.square(a - b))

def content_loss(session, model, content_image, layer_ids):
  feed_dict = model.create_feed_dict(image=content_image)
  layers = model.get_layer_tensors(layer_ids)

  values = session.run(layers, feed_dict=feed_dict)

  with model.graph.as_default():
    layer_losses = []

    # For each layer and its corresponding values
    # for the content-image.
    for value, layer in zip(values, layers):
      value_const = tf.constant(value)
      loss = mean_squared_error(layer, value_const)
      layer_losses.append(loss)

    total_loss = tf.reduce_mean(layer_losses)

  return total_loss

def gram_matrix(tensor):
  shape = tensor.get_shape()

  # Get the number of feature channels for the input tensor,
  # which is assumed to be from a convolutional layer with 4-dim.
  num_channels = int(shape[3])

  matrix = tf.reshape(tensor, shape=[-1, num_channels])

  return tf.matmul(tf.transpose(matrix), matrix)

def style_loss(session, model, style_image, layer_ids):
  feed_dict = model.create_feed_dict(image=style_image)

  layers = model.get_layer_tensors(layer_ids)

  with model.graph.as_default():
    gram_layers = [gram_matrix(layer) for layer in layers]
    values = session.run(gram_layers, feed_dict=feed_dict)

    layer_losses = []

    for value, gram_layer in zip(values, gram_layers):
      value_const = tf.constant(value)
      loss = mean_squared_error(gram_layer, value_const)
      layer_losses.append(loss)

    total_loss = tf.reduce_mean(layer_losses)

  return total_loss


# shifts input image by 1 pixel on x and y axis
# calculate difference between shifted and original image
# absolute value to make positive
# calculate sum of pixels in those images
# helps suppress noise in mixed image we are generating

def denoise_loss(model):
  loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
    tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))

  return loss
