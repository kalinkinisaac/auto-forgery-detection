import numpy as np
import matplotlib.pyplot as plt
from image_tools import to_jpeg
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from joblib import dump, load
import os, os.path
from PIL import Image
from image_tools import *
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def differences_image(image, quality=100, w=16):
    image_q1 = image
    image_q2 = to_jpeg(image_q1, quality)
    i_q1 = np.array(image_q1, dtype=float)
    i_q2 = np.array(image_q2, dtype=float)

    result = np.zeros(i_q1.shape)

    for x in range(i_q1.shape[0] - w):
        for y in range(i_q2.shape[1] - w):
            result[x, y] = np.sum((i_q1[x:x + w, y:y + w] - i_q2[x:x + w, y:y + w]) ** 2) / (3 * w ** 2)

    result *= 1 / result.max()
    result = np.power(result, 1 / 3)
    plt.imshow(result)
    plt.show()


def difference(i_q1: np.ndarray, i_q2: np.ndarray, x, y, w=8):
    return np.sum((i_q1[x:x + w, y:y + w] - i_q2[x:x + w, y:y + w]) ** 2) / (3 * w ** 2)


def difference_curve(image, x=0, y=0, w=16, q_min=30, q1=100):
    i_q1 = np.array(image, dtype=float)

    result = np.zeros(q1 - q_min + 1)
    for i in range(q1 - q_min + 1):
        i_q2 = np.array(to_jpeg(image, i + q_min), dtype=float)
        result[i] = difference(i_q1, i_q2, x, y, w)
    if result.max() != 0:
        result *= 1 / result.max()
    return result
    # plt.plot(np.arange(30, 101), result)
    # plt.show()


def get_curves(image, q_min=30, q1=100, w=8):
    n_cols, n_rows = np.array(image.size) // w
    x = n_cols * w
    y = n_rows * w

    i_q1 = np.array(
        image.crop((0, 0, x, y)),
        dtype=float
    )

    differences = np.zeros((q1 - q_min + 1, y, x))

    for q in range(q_min, q1 + 1):
        i_q2_cropped = to_jpeg(image, q).crop((0, 0, x, y))
        i_q2 = np.array(i_q2_cropped, dtype=float)
        differences[q - q_min] = np.sum((i_q1 - i_q2) ** 2, axis=2) / 3

    xw_sum = np.sum(differences.reshape((differences.shape[0], -1, w)), axis=2)
    yxw_sum = np.sum(
        np.transpose(
            xw_sum.reshape((differences.shape[0], -1, n_cols)),
            axes=(0, 2, 1)
        ).reshape((differences.shape[0], -1, w)),
        axis=2
    )
    yxw_sum = np.transpose(yxw_sum.reshape((differences.shape[0], -1, n_rows)), axes=(0, 2, 1))
    yxw_sum = yxw_sum / yxw_sum.max(axis=0)
    return yxw_sum

def predict_q1(image, q_min=30, w=8):
    # n_cols, n_rows = np.array(image.size) // w
    # x = n_cols * w
    # y = n_rows * w
    # image = image.crop((0, 0, x, y))
    # i_q1 = np.array(image, dtype=float)
    #
    # result = np.zeros((101 - q_min, x, y))
    #
    # for i in range(101 - q_min):
    #     i_q2 = np.array(to_jpeg(image, i + q_min), dtype=float)
    #     result[i] = np.sum((i_q1 - i_q2) ** 2) / (3 * np.product(i_q1.shape) ** 2)
    #
    # xw_sum = np.sum(result.reshape((result.shape[0], -1, w)), axis=2)
    # yxw_sum = np.sum(
    #     np.transpose(
    #         xw_sum.reshape((result.shape[0], -1, w)),
    #         axes=(0, 2, 1)
    #     ).reshape((result.shape[0], -1, w)),
    #     axis=2
    # )
    # yxw_sum = np.transpose(yxw_sum.reshape((result.shape[0], n_rows, n_cols)), axes=(0, 2, 1))
    yxw_sum = get_curves(image, q_min, w)
    minimums = np.argmin(yxw_sum, axis=0)

    return np.min(minimums) + q_min


def get_features(curve, q1, q_min=30):
    c = curve
    w1 = np.array([(x - q_min) / (q1 - q_min) for x in range(q_min, q1 + 1)])
    w2 = 1 - w1
    # Features:

    f1 = np.sum(np.dot(w1, c)) / np.sum(w1)

    f2 = np.median(c, axis=0)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(q_min, q1 + 1), c)
    f3 = slope
    f4 = intercept
    g5 = c.copy()
    t = 0.5
    g5[g5 < t] = 1
    g5[g5 != 1] = 0
    f5 = 1 / np.sum(w2) * np.sum(np.dot(w2, g5))
    g6 = w2 - c
    g6[g6 <= 0] = 0
    f6 = np.sum(np.square(g6))

    return f1, f2, f3, f4, f5, f6


def block_analysis(image, x, y, w=16, q_min=30):
    q1 = predict_q1(image)
    c = difference_curve(image, x, y, w)[:q1 - q_min + 1]
    return get_features(c, q1, q_min)


def data_preparation(path='', w=64):
    images = []
    valid_images = [".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        images.append(Image.open(os.path.join(path, f)).convert('RGB'))
    X = []
    Y = []
    for i, image in enumerate(images):
        print(f'image {i + 1} of {len(images)}')
        i_max = (image.size[1]) // w
        j_max = (image.size[0]) // w
        for _ in range(10):
            q0 = randint(40, 70)
            delta = randint(10, 20)
            q1 = q0 + delta

            single_compressed = to_jpeg(image, q0)
            double_compressed = to_jpeg(single_compressed, q1)
            single_yxw_sum = get_curves(single_compressed, q1=q1, w=w)
            double_yxw_sum = get_curves(double_compressed, q1=q1, w=w)

            for a in range(i_max - 1):
                for b in range(j_max - 1):
                    X.append(get_features(single_yxw_sum[:, a, b], q1))
                    Y.append(0)

            for a in range(i_max - 1):
                for b in range(j_max - 1):
                    X.append(get_features(double_yxw_sum[:, a, b], q1))
                    Y.append(1)

    np.save('X.npy', np.array(X))
    np.save('Y.npy', np.array(Y))



def ML():
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size=.8)

    def plot_roc_curve(model, X, y):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y, model.predict(X))
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate,
                 true_positive_rate,
                 'blue',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        print(model.score(X, y))
        print(false_positive_rate, true_positive_rate)

    clf = RandomForestClassifier(n_estimators=50, max_depth=2,
                                 random_state=0)
    clf.fit(x_train, y_train)
    # clf.fit(X, Y)
    plot_roc_curve(clf, x_test, y_test)
    dump(clf, 'model.joblib')

    # print(clf.predict([[0, 0, 0, 0]]))
def plot(image: Image):
    plt.imshow(np.array(image))
    plt.show()
def predict(image, w=64):
    clf = load('model.joblib')
    q1 = 84#predict_q1(image)
    curves = get_curves(image, q1=q1, w=w)
    for i in range(curves.shape[1]):
        for j in range(curves.shape[2]):
            f = get_features(curves[:, i, j], q1=q1)
            if clf.predict([f])[0] != 1:
                image = color_box(image, box=(j * w, i * w, j * w + w, i * w + w))
            else:
                image = color_box(image, box=(j * w, i * w, j * w + w, i * w + w), color=(0,255,0))
    plot(image)



