import argparse

from images import load_image, save_image

ap = argparse.ArgumentParser()

ap.add_argument(
  "--content_image",
  type=str,
  help="Name of the content image"
)
ap.add_argument(
  "--style_image",
  type=str,
  help="Name of the style image"
)
ap.add_argument(
  "--output_path",
  type=str,
  default="/output/" # for floydhub
  help="Path for output file"
)
ap.add_argument(
  "--model_path",
  type=str,
  default="vgg19/"
  help="Path for vgg19 network"
)
ap.add_argument(
  "--content_layers",
  nargs="+",
  type=str,
  default=["conv4_2"]
  help="VGG19 layers used for the content image"
)
ap.add_argument(
  "--style_layers",
  nargs="+",
  type=str,
  default=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
  help="VGG19 layers used for the content image"
)
ap.add_argument(
  "--content_layer_weights",
  nargs="+",
  type=str,
  default=[1.0]
  help="Contributions (weights) of each content layer to loss"
)
ap.add_argument(
  "--style_layer_weights",
  nargs="+",
  type=str,
  default=[0.2, 0.2, 0.2, 0.2, 0.2]
  help="Contributions (weights) of each style layer to loss"
)
ap.add_argument(
  "--content_loss_weight",
  type=float,
  default=5e0,
  help="Weight for the content loss function"
)
ap.add_argument(
  "--style_loss_weight",
  type=float,
  default=1e4,
  help="Weight for the style loss function"
)
ap.add_argument(
  "--tv_loss_weight",
  type=float,
  default=1e-3,
  help="Weight for the total variation loss function"
)
ap.add_argument(
  "--learning_rate",
  type=float,
  default=5,
  help="Learning rate for optimizers"
)
ap.add_argument(
  "--iterations",
  type=float,
  default=500,
  help="Number of iterations to run style transfer"
)
ap.add_argument(
  "--optimizer",
  type=str,
  default="adam",
  help="Optimizer for style transfer"
)
ap.add_argument(
  "--preserve_color",
  action="set_true",
  help="Transfer style but keep original content colors"
)

args = vars(ap.parse_args())

CONTENT_PATH = "data/content/"
STYLE_PATH = "data/styles/"
OUTPUT_PATH = "data/output/"

CONTENT_IMAGE_PATH = CONTENT_PATH + args['content_image']
STYLE_IMAGE_PATH = STYLE_PATH + args['style_image']

MODEL_PATH = args['model_path']
OUTPUT_PATH = args['output_path']
OUTPUT_IMAGE_PATH = OUTPUT_PATH + "output.jpg"

content_image = load_image(CONTENT_IMAGE_PATH, max_size=None)
style_image = load_image(STYLE_IMAGE_PATH, max_size=300)

CONTENT_LAYERS = args['content_layers']
STYLE_LAYERS = args['style_layers']
CONTENT_LAYER_WEIGHTS = args['content_layer_weights']
STYLE_LAYER_WEIGHTS = args['style_layer_weights']

CONTENT_LOSS_WEIGHT = args['content_loss_weight']
STYLE_LOSS_WEIGHT = args['style_loss_weight']
TV_LOSS_WEIGHT = args['tv_loss_weight']

LEARNING_RATE = args['learning_rate']
NUM_ITERATIONS = args['iterations']
OPTIMIZER = args['optimizer']

PRESERVE_COLOR = args['preserve_color']
