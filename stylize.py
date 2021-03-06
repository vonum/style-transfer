import argparse
import tensorflow as tf
import pickle

import vgg19
from style_transfer import StyleTransfer

from images import load_image, save_image, add_one_dim

ap = argparse.ArgumentParser()

ap.add_argument(
  "--content_image",
  "-c",
  type=str,
  help="Name of the content image"
)
ap.add_argument(
  "--style_image",
  "-s",
  type=str,
  help="Name of the style image"
)
ap.add_argument(
  "--output_image_path",
  "-o",
  type=str,
  default="/output/output.jpg", # for floydhub
  help="Path for output file"
)
ap.add_argument(
  "--loss_summary_path",
  "-sum",
  type=str,
  default="/output/loss_summary.pickle", # for floydhub
  help="Path for loss summary"
)
ap.add_argument(
  "--model_path",
  "-m",
  type=str,
  default="vgg19/vgg19.mat",
  help="Path for vgg19 network"
)
ap.add_argument(
  "--content_layers",
  "-cl",
  nargs="+",
  type=str,
  default=["conv4_2"],
  help="VGG19 layers used for the content image"
)
ap.add_argument(
  "--style_layers",
  "-sl",
  nargs="+",
  type=str,
  default=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
  help="VGG19 layers used for the content image"
)
ap.add_argument(
  "--content_layer_weights",
  "-clw",
  nargs="+",
  type=str,
  default=[1.0],
  help="Contributions (weights) of each content layer to loss"
)
ap.add_argument(
  "--style_layer_weights",
  "-slw",
  nargs="+",
  type=str,
  default=[0.2, 0.2, 0.2, 0.2, 0.2],
  help="Contributions (weights) of each style layer to loss"
)
ap.add_argument(
  "--content_loss_weight",
  "-cw",
  type=float,
  default=5e0,
  help="Weight for the content loss function"
)
ap.add_argument(
  "--style_loss_weight",
  "-sw",
  type=float,
  default=1e4,
  help="Weight for the style loss function"
)
ap.add_argument(
  "--tv_loss_weight",
  "-tvw",
  type=float,
  default=1e-3,
  help="Weight for the total variation loss function"
)
ap.add_argument(
  "--learning_rate",
  "-lr",
  type=float,
  default=5,
  help="Learning rate for optimizers"
)
ap.add_argument(
  "--iterations",
  "-it",
  type=int,
  default=500,
  help="Number of iterations to run style transfer"
)
ap.add_argument(
  "--optimizer",
  "-op",
  type=str,
  default="l_bfgs",
  help="Optimizer for style transfer"
)
ap.add_argument(
  "--max_size",
  type=int,
  default=512,
  help="Max size for content image"
)
ap.add_argument(
  "--init_type",
  type=str,
  default="content",
  help="How to initialize image (content | style | random)"
)
ap.add_argument(
  "--preserve_colors",
  "-pc",
  action="store_true",
  help="Transfer style but keep original content colors"
)
ap.add_argument(
  "--cvt_type",
  "-ct",
  type=str,
  default="ycrcb",
  help="How to transfer colors to result image (ycrcb | yuv | lab | luv)"
)
ap.add_argument(
  "--content_factor_type",
  "-cft",
  type=int,
  default=1,
  choices=[1, 2],
  help="Different types of normalization for content loss"
)
ap.add_argument(
  "--save_it",
  "-si",
  action="store_true",
  help="Save images through the style transfer process"
)
ap.add_argument(
  "--save_it_dir",
  "-sid",
  type=str,
  default=None,
  help="Directory in which to save images throgh the style transfer process"
)

args = vars(ap.parse_args())

CONTENT_PATH = "data/content/"
STYLE_PATH = "data/styles/"

CONTENT_IMAGE_PATH = CONTENT_PATH + args["content_image"]
STYLE_IMAGE_PATH = STYLE_PATH + args["style_image"]

MODEL_PATH = args["model_path"]
OUTPUT_IMAGE_PATH = args["output_image_path"]
LOSS_SUMMARY_PATH = args["loss_summary_path"]

MAX_SIZE = args["max_size"]

content_image = load_image(CONTENT_IMAGE_PATH, max_size=MAX_SIZE)
content_shape = [content_image.shape[1], content_image.shape[0]]
content_image = add_one_dim(content_image)
style_image = load_image(STYLE_IMAGE_PATH, shape=content_shape)
style_image = add_one_dim(style_image)

CONTENT_LAYERS = args["content_layers"]
STYLE_LAYERS = args["style_layers"]
CONTENT_LAYER_WEIGHTS = args["content_layer_weights"]
STYLE_LAYER_WEIGHTS = args["style_layer_weights"]

CONTENT_LOSS_WEIGHT = args["content_loss_weight"]
STYLE_LOSS_WEIGHT = args["style_loss_weight"]
TV_LOSS_WEIGHT = args["tv_loss_weight"]

LEARNING_RATE = args["learning_rate"]
ITERATIONS = args["iterations"]
OPTIMIZER = args["optimizer"]

INIT_TYPE = args["init_type"]
PRESERVE_COLORS = args["preserve_colors"]
CVT_TYPE = args["cvt_type"]
CONTENT_FACTOR_TYPE = args["content_factor_type"]

SAVE_IT = args["save_it"]
SAVE_IT_DIR = args["save_it_dir"]

net = vgg19.VGG19(MODEL_PATH)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

st = StyleTransfer(
  sess,
  net,
  ITERATIONS,
  CONTENT_LAYERS,
  STYLE_LAYERS,
  content_image,
  style_image,
  CONTENT_LAYER_WEIGHTS,
  STYLE_LAYER_WEIGHTS,
  CONTENT_LOSS_WEIGHT,
  STYLE_LOSS_WEIGHT,
  TV_LOSS_WEIGHT,
  OPTIMIZER,
  learning_rate=LEARNING_RATE,
  init_img_type=INIT_TYPE,
  preserve_colors=PRESERVE_COLORS,
  cvt_type=CVT_TYPE,
  content_factor_type=CONTENT_FACTOR_TYPE,
  save_it=SAVE_IT,
  save_it_dir=SAVE_IT_DIR
)

mixed_image = st.run()
summary = st.loss_summary()

sess.close()

save_image(mixed_image, OUTPUT_IMAGE_PATH)

with open(LOSS_SUMMARY_PATH, "wb") as handle:
  pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)
