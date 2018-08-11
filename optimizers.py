import tensorflow as tf

def create_optimizer(optimizer, learning_rate):
  if optimizer == "adam":
    return adam(learning_rate)
  elif optimizer == "gradient_descent":
    return gradient_descent(learning_rate)
  elif optimizer == "adagrad":
    return adagrad(learning_rate)
  else:
    raise "Unsupported optimizer"

def adam(learning_rate):
  return tf.train.AdamOptimizer(learning_rate=learning_rate)

def gradient_descent(learning_rate):
  return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

def l_bfgs():
  pass

def adagrad(learning_rate):
  return tf.train.AdagradOptimizer(learning_rate=learning_rate)
