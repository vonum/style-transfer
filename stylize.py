import argparse

from style_transfer import style_transfer
from images import load_image, save_image

ap = argparse.ArgumentParser()

ap.add_argument('--wc', type=float, default=5)
ap.add_argument('--ws', type=float, default=100)
ap.add_argument('--wd', type=float, default=0.1)
ap.add_argument('--lr', type=float, default=10)
ap.add_argument('--iter', type=int, default=500)
ap.add_argument('--opt', type=str, default="adam")
ap.add_argument('--content', type=str)
ap.add_argument('--style', type=str)
ap.add_argument('--model_path', type=str, default="vgg16/")

args = vars(ap.parse_args())

WEIGHT_CONTENT = args['wc']
WEIGHT_STYLE = args['ws']
WEIGHT_DENOISE = args['wd']
LEARNING_RATE = args['lr']
NUM_ITERATIONS = args['iter']
OPTIMIZER = args['opt']

CONTENT_PATH = "data/content/"
STYLE_PATH = "data/styles/"
OUTPUT_PATH = "data/output/"

CONTENT_IMAGE_PATH = CONTENT_PATH + args['content']
STYLE_IMAGE_PATH = STYLE_PATH + args['style']
OUTPUT_IMAGE_PATH = OUTPUT_PATH + "output.jpg"

MODEL_PATH = args['model_path']

content_image = load_image(CONTENT_IMAGE_PATH, max_size=None)
style_image = load_image(STYLE_IMAGE_PATH, max_size=300)

content_layer_ids = [4]
style_layer_ids = list(range(13))

mixed_image = style_transfer(content_image=content_image,
                             style_image=style_image,
                             content_layer_ids=content_layer_ids,
                             style_layer_ids=style_layer_ids,
                             weight_content=WEIGHT_CONTENT,
                             weight_style=WEIGHT_STYLE,
                             weight_denoise=WEIGHT_DENOISE,
                             learning_rate=LEARNING_RATE,
                             num_iterations=NUM_ITERATIONS,
                             optimizer=OPTIMIZER,
                             data_dir=MODEL_PATH)

save_image(mixed_image, OUTPUT_IMAGE_PATH)
