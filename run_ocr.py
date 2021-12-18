import pytesseract
from PIL import Image
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())

image = Image.open(args['image'])

print(pytesseract.image_to_string(image))