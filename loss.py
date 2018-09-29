import tensorflow as tf

def mean_squared_error(a, b):
  return tf.reduce_mean(tf.square(a - b))

def sum_squared_error(a, b):
  return tf.reduce_sum(tf.square(a - b))

def gram_matrix(tensor):
  shape = tensor.get_shape()

  # Get the number of feature channels for the input tensor,
  # which is assumed to be from a convolutional layer with 4-dim.
  num_channels = int(shape[3])

  matrix = tf.reshape(tensor, shape=[-1, num_channels])

  return tf.matmul(tf.transpose(matrix), matrix)

# helps suppress noise in mixed image we are generating
def tv_loss(image):
  return tf.image.total_variation(image)
