import tensorflow as tf

L_BFGS = "l_bfgs"
ADAM = "ADAM"
ADAGRAD = "ADAGRAD"
GRADIENT_DESCENT = "GRADIENT_DESCENT"

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

def l_bfgs(loss, max_iter):
  return tf.contrib.opt.ScipyOptimizerInterface(
    loss,
    method="L-BFGS-B",
    options={"maxiter": max_iter}
  )

def adagrad(learning_rate):
  return tf.train.AdagradOptimizer(learning_rate=learning_rate)
