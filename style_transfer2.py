import tensorflow as tf
import numpy as np

from loss import gram_matrix, sum_squared_error, tv_loss
import optimizers
from optimizers import l_bfgs, adam, adagrad, gradient_descent
from images import plot_images

INIT_IMG_RANDOM = "random"
INIT_IMG_CONTENT = "content"
INIT_IMG_STYLE = "style"

# add content loss norm type and init strategy
class StyleTransfer:
  def __init__(self, sess, net, iterations, content_layers, style_layers,
               content_image, style_image, content_layer_weights,
               style_layer_weights, content_loss_weight, style_loss_weight,
               tv_loss_weight, optimizer_type, learning_rate=None,
               plot=False, init_img_type=INIT_IMG_RANDOM):
    self.sess = sess
    self.net = net
    self.iterations = iterations
    self.optimizer_type = optimizer_type
    self.learning_rate = learning_rate

    self.init_img_type = init_img_type
    self.plot = plot

    self.content_layers = content_layers
    self.style_layers = style_layers
    self.content_layer_weights = content_layer_weights
    self.style_layer_weights = style_layer_weights

    self.alpha = content_loss_weight
    self.beta = style_loss_weight
    self.theta = tv_loss_weight

    # variable names from the paper
    self.p0 = np.float32(self._preprocess_image(content_image))
    self.a0 = np.float32(self._preprocess_image(style_image))
    self.x0 = self.init_img()

    self._build_graph()

  def run(self):
    self.sess.run(tf.global_variables_initializer())

    self._optimize()

    res_img = self.sess.run(self.x)
    res_img = self._postprocess_image(res_img)

    return res_img

  def _build_graph(self):
    # result image
    self.x = tf.Variable(self.x0, trainable=True, dtype=tf.float32)

    self.p = tf.placeholder(tf.float32, shape=self.p0.shape, name="content")
    self.a = tf.placeholder(tf.float32, shape=self.a0.shape, name="style")

    self._create_content_activations()
    self._create_style_activations()
    self._create_mixed_image_activations()

    self._create_loss()

    self._create_optimizer()

  def _create_content_activations(self):
    content_activations = self._feed_forward(self.p, scope="content")
    # activations for content image for specified layers
    self.Ps = [content_activations[id] for id in self.content_layers]

  def _create_style_activations(self):
    style_activations = self._feed_forward(self.a, scope="style")
    # activations for style image for specified layers
    self.As = [style_activations[id] for id in self.style_layers]

  def _create_mixed_image_activations(self):
    Xs = self._feed_forward(self.x, scope="mixed")
    self.Xs_content = [Xs[id] for id in self.content_layers]
    self.Xs_style = [Xs[id] for id in self.style_layers]

  def _create_loss(self):
    self._create_content_loss()
    self._create_style_loss()
    self._create_tv_loss()

    self.loss = self.alpha * self.content_loss + \
            self.beta * self.style_loss + \
            self.theta * self.tv_loss

  def _create_content_loss(self):
    self.content_loss = 0
    cw = self.content_layer_weights

    for i in range(0, len(self.content_layers)):
      X = self.Xs_content[i]
      P = self.Ps[i]

      # batch_size, height, width, number of filters
      _, h, w, d = X.get_shape()
      N = h.value * w.value
      M = d.value
      factor = (1. / (2. * np.sqrt(M) * np.sqrt(N)))
      self.content_loss += cw[i] * sum_squared_error(X, P) * factor

  def _create_style_loss(self):
    self.style_loss = 0
    sw = self.style_layer_weights

    for i in range(0, len(self.style_layers)):
      X = self.Xs_style[i]
      A = self.As[i]

      # batch_size, height, width, number of filters
      _, h, w, d = X.get_shape()
      N = h.value * w.value
      M = d.value

      X_gram = gram_matrix(X)
      A_gram = gram_matrix(A)

      sse = sum_squared_error(X_gram, A_gram)
      self.style_loss += sw[i] * (1. / (4 * N ** 2 * M ** 2)) * sse

  def _create_tv_loss(self):
    self.tv_loss = tv_loss(self.x[0])

  def _create_optimizer(self):
    if self.optimizer_type == optimizers.L_BFGS:
      self.optimizer = l_bfgs(self.loss, self.iterations)
    elif self.optimizer_type == optimizers.ADAM:
      self.optimizer = adam(self.learning_rate)
      self.train_op = self.optimizer.minimize(self.loss)
    elif self.optimizer_type == optimizers.ADAGRAD:
      self.optimizer = adagrad(self.learning_rate)
      self.train_op = self.optimizer.minimize(self.loss)
    elif self.optimizer_type == optimizers.GRADIENT_DESCENT:
      self.optimizer = gradient_descent(self.learning_rate)
      self.train_op = self.optimizer.minimize(self.loss)
    else:
      raise "Unsupported optimizer"

  def _optimize(self):
    if self.optimizer_type == optimizers.L_BFGS:
      self._optimize_lbgfs()
    else:
      self._optimize_rest()

  def _optimize_lbgfs(self):
    global _iter
    _iter = 0

    def callback(x, l, cl, sl, tvl):
      global _iter
      self._print_training_state(_iter, x, l, cl, sl, tvl)
      _iter += 1

    feed_dict = {self.p: self.p0, self.a: self.a0}
    fetches = [self.x, self.loss, self.content_loss, self.style_loss, self.tv_loss]

    self.optimizer.minimize(
      self.sess,
      feed_dict=feed_dict,
      fetches=fetches,
      loss_callback=callback
    )

  def _optimize_rest(self):
    feed_dict = {self.p: self.p0, self.a: self.a0}
    run_list = [self.train_op, self.x, self.loss,
                self.content_loss, self.style_loss, self.tv_loss]

    for i in range(0, self.iterations):
      _, x, l, cl, sl, tvl = self.sess.run(run_list,
                                           feed_dict=feed_dict)
      self._print_training_state(i, x, l, cl, sl, tvl)

  def init_img(self):
    if self.init_img_type == INIT_IMG_RANDOM:
      return np.random.normal(size=self.p0.shape, scale=np.std(self.p0))
    elif self.init_img_type == INIT_IMG_CONTENT:
      return self.p0
    elif self.init_img_type == INIT_IMG_STYLE:
      return self.a0
    else:
      raise "Unsupported initialization strategy"

  def _content_norm_factor(self, strategy):
    pass

  def _preprocess_image(self, image):
    return self.net.preprocess(image)

  def _postprocess_image(self, image):
    res_image = np.clip(self.net.undo_preprocess(image), 0.0, 255.0)
    # remove the batch dimension
    shape = res_image.shape
    return np.reshape(res_image, shape[1:])

  def _plot_images(self, p, x, a):
    p = self._postprocess_image(p)
    a = self._postprocess_image(a)
    x = self._postprocess_image(x)
    plot_images(content_image=p, style_image=a, mixed_image=x)

  def _feed_forward(self, tensor, scope=None):
    return self.net.feed_forward(tensor, scope)

  def _print_training_state(self, i, x, l, cl, sl, tvl):
    if (i % 10 == 0) or (i == self.iterations - 1):
      print(f"Iteration: {i}|loss {l}|{cl}|{sl}|{tvl}")
      if self.plot: self._plot_images(self.p0, x, self.a0)
