import math

import tensorflow as tf


def make_grid(tensor,
              nrow: int = 8,
              padding: int = 2,
              pad_value: int = 0):
    """make the mini-batch of images into a grid"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    num_channels = tensor.shape[3]
    grid = tf.fill([height * ymaps + padding, width * xmaps + padding, num_channels],
                   tf.constant(pad_value, tensor.dtype))
    grid = tf.Variable(grid)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            placeholder = grid[y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width]
            placeholder.assign(tf.identity(tensor[k]))
            k += 1
    return grid.read_value()
