import gen_data_array
import numpy as np
import pandas as pd
from skimage.transform import pyramid_gaussian
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from skimage import io, color
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math as m


def main():
    try:
        model = gen_data_array.open_pickle("HOG_svm_model.pkl")
    except:
        train_HOG_model()
        model = gen_data_array.open_pickle("HOG_svm_model.pkl")
    while True:
        file_name = input("Please enter image filename: ")
        im = io.imread(file_name)
        im = color.rgb2gray(im)
        im = np.around(255 * im).astype(np.uint8)

        comp = get_im_pyramid(im)
        pedestrian_scan(model, comp, (128, 64))


def pedestrian_scan(model, comp_img, window):
    #Probably throw an assert in here
    w_r, w_c = window
    x = np.zeros(1000)
    y = np.zeros(1000)
    k = 0
    for r in range(0, comp_img.shape[0] - w_r + 1, 2):
        for c in range(0, comp_img.shape[1] - w_c + 1, 2):
            bound = comp_img[r: r + w_r, c: c + w_c]
            val = model.predict(hog(bound, 9, (8, 8), (2, 2), visualize=False, feature_vector=True).reshape(1, -1))
            if val == 1:
                x[k] = c
                y[k] = r
                k += 1
    fig, ax = plt.subplots(1)
    ax.imshow(comp_img)
    for j in range(k + 1):
        rect = patches.Rectangle((x[j], y[j]), 64, 128, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def get_im_pyramid(image):
    rows, cols = image.shape
    pyramid = tuple(pyramid_gaussian(image, max_layer=(m.floor(m.log(rows, 2)-2)), downscale=2, multichannel=False))
    composite_image = np.zeros((rows, cols + cols // 2 + 1))
    composite_image[:rows, :cols] = pyramid[0]
    i_row = 0

    for p in pyramid[1:]:
        n_rows, n_cols = p.shape
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows

    return composite_image
    #plt.imshow(composite_image)
    #plt.axis('off')
    #plt.show()



def train_HOG_model():
    try:
        data = gen_data_array.open_pickle("transformed_set.pkl")
        X_f, y = data[:, :-1], data[:, -1]
        print(X_f.shape)
        print(y.shape)
    except:
        X, y = make_pedestrians()
        X_f = HOG_transform(X, (128, 64), 9, (8, 8), (2, 2))
        print(X_f.shape[0])
        gen_data_array.save_pickle("transformed_set.pkl", np.concatenate((X_f, y), axis=1))
    df = pd.DataFrame(data)
    train_set, test_set = split_train_test(df, 0.2)

    poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
    poly_kernel_svm_clf.fit(train_set.values[:, :-1], train_set.values[:, -1])
    print(poly_kernel_svm_clf.score(test_set.values[:, :-1], test_set.values[:, -1]))
    gen_data_array.save_pickle("HOG_svm_model.pkl", poly_kernel_svm_clf)


def split_train_test(data, test_ratio):
    """Copied from Aurelien Geron's Hands-on ML book

    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


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
