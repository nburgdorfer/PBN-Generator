import numpy as np
import sys
import os
import yaml
import argparse
import cv2
import matplotlib.pyplot as plt

## Argument Parsing ##
parse = argparse.ArgumentParser(description="A simple Paint-By-Numbers image generator.")
parse.add_argument("-c", "--color_palette", default="palettes/earthy.yaml", type=str, help="Color palette file to use for the image.")
parse.add_argument("-i", "--image", default="palettes/earthy.yaml", type=str, help="The image file to use for the canvas.")
parse.add_argument("-w", "--width", default=-1, type=int, help="Desired width of output canvas.")
parse.add_argument("-t", "--height", default=-1, type=int, help="Desired height of output canvas.")
#parse.add_argument("-f", "--flag", action="store_true", help="")
ARGS = parse.parse_args()

def display_palette(color_names, color_codes):
    print("Current color palette:")
    for n,c in zip(color_names, color_codes):
        print("{}: #{}".format(n,c))

def segment_image(image, color_codes):
    k = len(color_codes)
    pixels = np.float32(image.reshape((-1,3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))

    return cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

def load_image(img_file, width, height):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop_w, crop_h, _ = img.shape

    if(width != -1):
        crop_w = min(width, crop_w)
    if(height != -1):
        crop_h = min(height, crop_h)

    return img[:crop_w,:crop_h,:]


def get_palette(palette_file):
    with open(palette_file, 'r') as pf:
        palette = yaml.safe_load(pf)
        names = palette["names"]
        codes = palette["codes"]

        assert len(names) == len(codes), "There was an error loading the color palette."

        num_colors = len(names)
        
        return (names, codes)

def main():
    (color_names, color_codes) = get_palette(ARGS.color_palette)

    #display_palette(color_names,color_codes)
    image = load_image(ARGS.image, ARGS.width, ARGS.height)

    seg_image = segment_image(image, color_codes)


    cv2.imwrite("output.png", seg_image)


if __name__=="__main__":
    main()

