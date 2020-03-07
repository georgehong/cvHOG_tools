"""Image Training Files to Single Array

This script allows the user to access all image files in a directory and
samples a random window of constant size from each file.  If all images are the size
of the window, the entire image is saved.  The images are saved in grayscale and are
the rows of a numpy array.  This numpy array is saved via pickle.

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
    num_files = int(input("Please enter an upper bound for the number of training files: "))
    out_name = input("Please enter name of target pickle file: ")

    out = get_array(directory, (y_in, x_in), num_files)
    save_pickle(out_name, out)


def save_pickle(name, contents):
    """Convenience function for saving object

    Parameters
    __________
    name : str
        name of pickle file to save to.
    contents : obj
        object to save
    """
    with open(name, "wb") as f:
        pickle.dump(contents, f)


def open_pickle(name):
    """Convenience function for extracting object

    Parameters
    __________
    name : str
        name of pickle file to load object from

    Returns
    _______
    x : object
        object stored in the pickle file
    """
    with open(name, "rb") as f:
        x = pickle.load(f)
    return x


def get_array(path, window_sz, num_files):
    """Combines random windows of image files

    Parameters
    __________
    path : str
        path to folder containing all the images.
    window_sz : int tuple
        (height, width) of sampling window.
    num_files : int
        Upper bound for the number of images contained.

    Returns
    _______
    out
        numpy array of dimension num_files x window_sz area.  Rows are samples.
    """
    y, x = window_sz
    instances = 0

    out = np.zeros((num_files, x * y))
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
