import tensorflow as tf
import numpy as np

from loss import gram_matrix
import optimizers
from optimizers import l_bfgs
from images import plot_images

# add content loss norm type and init strategy
class StyleTransfer:
  def __init__(self, sess, net, iterations, content_layers, style_layers,
               content_image, style_image, content_layer_weights,
               style_layer_weights, content_loss_weight, style_loss_weight,
               tv_loss_weight, optimizer_type, learning_rate=None,
               plot=False):
    self.sess = sess
    self.net = net
    self.iterations = iterations
    self.optimizer_type = optimizer_type
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
    self.x0 = self.init_img(content_image, "")

  def run(self):
    self._build_graph()

    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

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
      self.content_loss += cw[i] * tf.reduce_sum(tf.square(X - P)) / 2

    return self.content_loss

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

      self.style_loss += sw[i] * (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.square(X_gram - A_gram))

    return self.style_loss

  def _create_tv_loss(self):
    self.tv_loss = tf.image.total_variation(self.x[0])

  def _create_optimizer(self):
    if self.optimizer_type == optimizers.L_BFGS:
      self.optimizer = l_bfgs(self.loss, self.iterations)
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
      if (_iter % 10 == 0) or (_iter == self.iterations - 1):
        print(f"Iteration: {_iter}|loss {l}|{cl}|{sl}|{tvl}")
        if self.plot: self._plot_images(self.p0, x, self.a0)
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
    pass

  def init_img(self, image, strategy):
    return np.random.normal(size=image.shape, scale=np.std(image))

  def _content_norm_factor(self, strategy):
    pass

  def _preprocess_image(self, image):
    return self.net.preprocess(image)

  def _postprocess_image(self, image):
    return np.clip(self.net.undo_preprocess(image), 0.0, 255.0)[0]

  def _plot_images(self, p, x, a):
    p = self._postprocess_image(p)
    a = self._postprocess_image(a)
    x = self._postprocess_image(x)
    plot_images(content_image=p, style_image=a, mixed_image=x)

  def _feed_forward(self, tensor, scope=None):
    return self.net.feed_forward(tensor, scope)
