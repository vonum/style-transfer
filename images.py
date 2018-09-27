import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

def load_image(filename, max_size=None, shape=None):
  # PIL.Image.LANCZOS is one of resampling filter
  image = PIL.Image.open(filename)

  if max_size is not None:
     factor = max_size / np.max(image.size)

     # Scale the image's height and width.
     size = np.array(image.size) * factor
     size = size.astype(int)

     image = image.resize(size, PIL.Image.LANCZOS)

  if shape is not None:
    image = image.resize(shape, PIL.Image.LANCZOS)

  return np.float32(image)

# VGG19 requires input dimension to be (batch, height, width, channel)
def add_one_dim(image):
  shape = (1,) + image.shape
  return np.reshape(image, shape)

def save_image(image, filename):
  # Ensure the pixel-values are between 0 and 255.
  image = np.clip(image, 0.0, 255.0)
  image = image.astype(np.uint8)

  with open(filename, 'wb') as file:
    PIL.Image.fromarray(image).save(file, 'jpeg')

def image_big(image):
  image = np.clip(image, 0.0, 255.0)
  image = image.astype(np.uint8)

  return PIL.Image.fromarray(image)

def plot_images(content_image, style_image, mixed_image):
  # Create figure with sub-plots.
  fig, axes = plt.subplots(1, 3, figsize=(10, 10))

  # Adjust vertical spacing.
  fig.subplots_adjust(hspace=0.1, wspace=0.1)

  # Use interpolation to smooth pixels?
  smooth = True

  if smooth:
    interpolation = 'sinc'
  else:
    interpolation = 'nearest'

  # Plot the content-image.
  # Note that the pixel-values are normalized to
  # the [0.0, 1.0] range by dividing with 255.
  ax = axes.flat[0]
  ax.imshow(content_image / 255.0, interpolation=interpolation)
  ax.set_xlabel("Content")

  # Plot the mixed-image.
  ax = axes.flat[1]
  ax.imshow(mixed_image / 255.0, interpolation=interpolation)
  ax.set_xlabel("Mixed")

  # Plot the style-image
  ax = axes.flat[2]
  ax.imshow(style_image / 255.0, interpolation=interpolation)
  ax.set_xlabel("Style")

  # Remove ticks from all the plots.
  for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

  # Ensure the plot is shown correctly with multiple plots
  # in a single Notebook cell.
  plt.show()
