import cv2
import numpy as np
from PIL import Image

class ColorTransfer:
  # content_img - image containing desired content
  # color_img - image containing desired color
  def __init__(self, content_img, color_img):
    self.content_img = content_img
    self.color_img = color_img

  def luminance_transfer(self, convert_type):
    content_img  = self.content_img
    color_img = self.color_img

    if convert_type == "yuv":
      cvt_type = cv2.COLOR_BGR2YUV
      inv_cvt_type = cv2.COLOR_YUV2BGR
    elif convert_type == "ycrcb":
      cvt_type = cv2.COLOR_BGR2YCR_CB
      inv_cvt_type = cv2.COLOR_YCR_CB2BGR
    elif convert_type == "luv":
      cvt_type = cv2.COLOR_BGR2LUV
      inv_cvt_type = cv2.COLOR_LUV2BGR
    elif convert_type == "lab":
      cvt_type = cv2.COLOR_BGR2LAB
      inv_cvt_type = cv2.COLOR_LAB2BGR

    content_cvt = self._convert(content_img, cvt_type)
    color_cvt = self._convert(color_img, cvt_type)

    c1, _, _ = self._split_channels(content_cvt)
    _, c2, c3 = self._split_channels(color_cvt)

    img = self._merge_channels([c1, c2, c3])
    img = self._convert(img, inv_cvt_type).astype(np.float32)

    return img

  def _split_channels(self, image):
    return cv2.split(image)

  def _merge_channels(self, channels):
    return cv2.merge(channels)

  def _convert(self, img, cvt_type):
    return cv2.cvtColor(img, cvt_type)
