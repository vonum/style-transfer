import tensorflow as tf

L_BFGS = "l_bfgs"
ADAM = "adam"
ADAGRAD = "adagrad"
GRADIENT_DESCENT = "gradient_descent"
RMSPROP = "rmsprop"

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

def rmsprop(learning_rate):
  return tf.train.RMSPropOptimizer(learning_rate=learning_rate)
