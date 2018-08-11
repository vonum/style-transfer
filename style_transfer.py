import tensorflow as tf
import numpy as np

import vgg16
from loss import content_loss, style_loss, denoise_loss
from images import plot_images

def style_transfer(content_image, style_image,
                   content_layer_ids, style_layer_ids,
                   weight_content=1.5, weight_style=10.0,
                   weight_denoise=0.3, learning_rate=10.0,
                   num_iterations=120):

  model = vgg16.VGG16()
  session = tf.InteractiveSession(graph=model.graph)

  # Print the names of the content-layers.
  print("Content layers:")
  print(model.get_layer_names(content_layer_ids))
  print()

  # Print the names of the style-layers.
  print("Style layers:")
  print(model.get_layer_names(style_layer_ids))
  print()

  # Create the loss-function for the content-layers and -image.
  loss_content = content_loss(session=session,
                              model=model,
                              content_image=content_image,
                              layer_ids=content_layer_ids)

  # Create the loss-function for the style-layers and -image.
  loss_style = style_loss(session=session,
                          model=model,
                          style_image=style_image,
                          layer_ids=style_layer_ids)

  # Create the loss-function for the denoising of the mixed-image.
  loss_denoise = denoise_loss(model)

  #adjust levels of loss functions, normalize them
  #multiply them with a variable
  #taking reciprocal values of loss values of content, style, denoising
  #small constant to avoid divide by 0
  #adjustment value normalizes loss so approximately 1
  #weights should be set relative to each other dont depend on layers
  #we are using

  # Create TensorFlow variables for adjusting the values of
  # the loss-functions. This is explained below.
  adj_content = tf.Variable(1e-10, name='adj_content')
  adj_style = tf.Variable(1e-10, name='adj_style')
  adj_denoise = tf.Variable(1e-10, name='adj_denoise')

  # Initialize the adjustment values for the loss-functions.
  session.run([adj_content.initializer,
               adj_style.initializer,
               adj_denoise.initializer])

  # Create TensorFlow operations for updating the adjustment values.
  # These are basically just the reciprocal values of the
  # loss-functions, with a small value 1e-10 added to avoid the
  # possibility of division by zero.
  update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
  update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
  update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

  loss_combined = weight_content * adj_content * loss_content + \
                  weight_style * adj_style * loss_style + \
                  weight_denoise * adj_denoise * loss_denoise

  optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
  gvs = optimizer.compute_gradients(loss_combined, model.input)

  # List of tensors that we will run in each optimization iteration.
  run_list = [gvs, update_adj_content, update_adj_style, update_adj_denoise]

  # The mixed-image is initialized with random noise.
  # It is the same size as the content-image.
  #where we first init it
  mixed_image = np.random.rand(*content_image.shape) + 128

  for i in range(num_iterations):
    # Create a feed-dict with the mixed-image.
    feed_dict = model.create_feed_dict(image=mixed_image)

    # Use TensorFlow to calculate the value of the
    # gradient, as well as updating the adjustment values.
    gvs, adj_content_val, adj_style_val, adj_denoise_val = session.run(run_list, feed_dict=feed_dict)

    grad = [gradient[0] for gradient in gvs]

    # Reduce the dimensionality of the gradient.
    #Remove single-dimensional entries from the shape of an array.
    grad = np.squeeze(grad)

    # Scale the step-size according to the gradient-values.
    #Ratio of weights:updates
    #akin to learning rate
    # step_size_scaled = step_size / (np.std(grad) + 1e-8)

    # Update the image by following the gradient.
    #gradient descent
    # mixed_image -= grad * step_size_scaled
    mixed_image -= grad / (np.std(grad) + 1e-8)

    # Ensure the image has valid pixel-values between 0 and 255.
    #Given an interval, values outside the interval are clipped
    #to the interval edges.
    mixed_image = np.clip(mixed_image, 0.0, 255.0)

    # Print a little progress-indicator.
    print(". ", end="")

    # Display status once every 10 iterations, and the last.
    if (i % 10 == 0) or (i == num_iterations - 1):
      print()
      print("Iteration:", i)

      # Print adjustment weights for loss-functions.
      msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
      print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

      #in larger resolution
      # Plot the content-, style- and mixed-images.
      plot_images(content_image=content_image,
                  style_image=style_image,
                  mixed_image=mixed_image)

  # Close the TensorFlow session to release its resources.
  session.close()

  # Return the mixed-image.
  return mixed_image
