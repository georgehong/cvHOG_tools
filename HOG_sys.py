import gen_data_array
import numpy as np
from skimage.transform import pyramid_gaussian
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def main():
    try:
        data = gen_data_array.open_pickle("transformed_set.pkl")
        X_f, y = data[:, :-1], data[:, 0]
        print(X_f.shape)
        print(y.shape)
    except:
        X, y = make_pedestrians()
        X_f = HOG_transform(X, (128, 64), 9, (8, 8), (2, 2))
        print(X_f.shape[0])
        gen_data_array.save_pickle("transformed_set.pkl", np.concatenate((X_f, y), axis=1))


def nparr_to_pd




def HOG_transform(X, image_shape, orientations, pixels_per_cell=(16, 16), cells_per_block=(1, 1)):
    """Converts dataset into HOG features with the entered parameters

    Parameters
    __________
    X : input dataset, entries are rows
    """
    X_out = [hog(x.reshape(image_shape), orientations,
                pixels_per_cell, cells_per_block, visualize=False, feature_vector=True) for x in X]
    return np.array(X_out)


def make_pedestrians():
    """Get data set of 128 x 64 images.  Data is unshuffled.

    Returns
    _______
    X : numpy array
        samples correspond to rows
    y : numpy array
        label
    """
    # Construct the shape of the data
    x_pos = gen_data_array.open_pickle("set_pos.pkl")
    y_pos = np.ones((x_pos.shape[0], 1))
    x_neg = gen_data_array.open_pickle("set_neg.pkl")
    y_neg = np.zeros((x_neg.shape[0], 1))

    y = np.concatenate((y_pos, y_neg), axis=0)
    X = np.concatenate((x_pos, x_neg), axis=0)
    return X, y


if __name__ == "__main__":
    main()
