"""Image Training Files to Single Array

This script allows the user to access all image files in a directory and
samples a random window of constant size from each file.  If all images are the size
of the window, the entire image is saved.  The images are saved in grayscale and are
the rows of a numpy array.  This numpy array is saved via pickle.

This script supports a maximum of 20000 images.

This file can be imported as a module and contains the following functions:

    * get_array - returns the array of image windows
    * open_pickle - shortcut to retrieve array in pickle file
    * save_pickle - convenience function to store array in pickle file
"""

import numpy as np
from skimage import io, color
import pickle
import os


def main():
    directory = input("Please enter path of directory containing all of the images: ")
    y_in = int(input("Please enter window height: "))
    x_in = int(input("Please enter window length: "))
    out_name = input("Please enter name of target pickle file: ")

    out = get_array(directory, (y_in, x_in))
    save_pickle(out_name, out)


def save_pickle(name, contents):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(contents, f)


def open_pickle(name):
    with open(name + ".pkl", "rb") as f:
        x = pickle.load(f)
    return x

def get_array(path, window_sz):
    # TODO finish comments
    """Combines random windows of image files

    Parameters
    __________
    path : str

    """
    y, x = window_sz
    instances = 0
    # With the given image sets, it's sometimes unclear how many of the images are the proper size
    # to generate a window with.  This is a very generous estimate.
    out = np.zeros((20000, x * y))
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                im = io.imread(file_path)
                im = color.rgb2gray(im)
                im = np.around(255 * im).astype(np.uint8)
                assert im.shape[0] >= y and im.shape[1] >= x
            except:
                continue

            print("Progress: " + str(instances))
            y_up, x_up = im.shape - np.array([y, x])  # Compute upper bound for sampling
            y0 = np.random.randint(0, y_up + 1)
            yf = y0 + y
            x0 = np.random.randint(0, x_up + 1)
            xf = x0 + x
            out[instances, :] = im[y0:yf, x0:xf].flatten()
            instances += 1

    return out[:instances, :]

if __name__ == "__main__":
    main()