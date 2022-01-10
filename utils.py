import numpy as np
import sklearn.cluster
import sklearn.metrics.cluster
import sklearn.metrics.pairwise
import tensorflow as tf


def parse_image(filename, image_size, minmax=(-1, 1)):
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (minmax[1] - minmax[0]) * image + minmax[0]
    shape = tf.shape(image)
    # center cropping
    h, w = shape[-3], shape[-2]
    if h > w:
        cropped_image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
    else:
        cropped_image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)

    image = tf.image.resize(cropped_image, image_size)

    return image


def smooth_labels(label, label_smoothing=0.1):
    nclass = tf.shape(label)[-1]
    return label * (1 - label_smoothing) + (label_smoothing / tf.cast(nclass - 1, tf.float32))


def nca_loss(y_true, y_pred):
    y_true_smoothed = smooth_labels(y_true)
    loss = tf.math.reduce_sum(-y_true_smoothed * tf.nn.log_softmax(-y_pred), axis=-1)
    return tf.math.reduce_mean(loss)


def cluster_by_kmeans(X, clusters):
    return sklearn.cluster.KMeans(clusters).fit(X).labels_


def compute_normalized_mutual_information(ys, xs_clustered):
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys)


def assign_by_euclidean_at_k(X, T, k):
    distances = sklearn.metrics.pairwise.pairwise_distances(X)
    indices = np.argsort(distances, axis=1)[:, 1:k + 1]
    return np.array([[T[i] for i in ii] for ii in indices])


def compute_recall_at_k(T, Y, k):
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))
