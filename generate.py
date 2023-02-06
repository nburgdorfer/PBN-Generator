import numpy as np
import sys
import os
import yaml
import argparse
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

## Argument Parsing ##
parse = argparse.ArgumentParser(description="A simple Paint-By-Numbers image generator.")
parse.add_argument("-c", "--color_palette", default="palettes/earthy.yaml", type=str, help="Color palette file to use for the image.")
parse.add_argument("-i", "--image", default="palettes/earthy.yaml", type=str, help="The image file to use for the canvas.")
parse.add_argument("-k", "--filter_size", default=11, type=int, help="The filter size for mode filtering.")
ARGS = parse.parse_args()

def display_palette(color_names, color_codes):
    print("Current color palette:")
    for n,c in zip(color_names, color_codes):
        print("{}: {}".format(n,c))

def segment_image(image, color_codes):
    k = len(color_codes)
    pixels = np.float32(image.reshape((-1,3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))

    return cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

def load_image(img_file):
    scale = 0.25
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def get_palette(palette_file):
    with open(palette_file, 'r') as pf:
        palette = yaml.safe_load(pf)
        names = palette["names"]
        codes = palette["codes"]

        assert len(names) == len(codes), "There was an error loading the color palette."

        num_colors = len(names)
        
        return (names, codes)

def draw_borders(image):
    rows,cols,c = image.shape
    b_img = np.zeros((rows,cols,c))

    for r in range(1,rows-1):
        for c in range(1,cols-1):
            patch = image[r-1:r+2,c-1:c+2,:].flatten().reshape(9,3)
            c_center = image[r,c].reshape(1,1,3)
            diff = np.where(patch != c_center, 1, 0)
            print(diff)

            if (diff > 0):
                b_img[r,c] = np.array([0,0,0])
            else:
                #b_img[r,c] = c_center
                b_img[r,c] = np.array([255,255,255])

    return b_img

def smooth(image, k=9):
    image = image / np.max(image)
    img = Image.fromarray(np.uint8(image*255))
    filt_image = img.filter(ImageFilter.ModeFilter(size = k)) 
    return np.asarray(filt_image)

def main():
    (color_names, color_codes) = get_palette(ARGS.color_palette)

    print("Loading image...")
    image = load_image(ARGS.image)
    cv2.imwrite("input.png", image)

    print("Segmenting image...")
    image = segment_image(image, color_codes)
    cv2.imwrite("seg.png", image)

    print("Smoothing image...")
    image = smooth(image, k=ARGS.filter_size)
    cv2.imwrite("filt.png", image)


    print("Drawing borders")
    image = draw_borders(image)

    cv2.imwrite("pbn.png", image)


if __name__=="__main__":
    main()

