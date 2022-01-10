from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from net.resnet_proxy_nca import ResnetProxyNCA
from utils import (
    parse_image,
    nca_loss,
    cluster_by_kmeans,
    compute_normalized_mutual_information,
    assign_by_euclidean_at_k,
    compute_recall_at_k,
)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataroot', default='dataset/celeba/', type=Path, help='path to dataset')
parser.add_argument('--batch-size', default=1000, type=int, help='input batch size')
parser.add_argument('--epochs', default=500, type=int, help='number of training epochs')
parser.add_argument('--image-size', default=64, type=int, help='the height / width of the input image to network')
parser.add_argument('--nch', default=3, type=int, help='size of input channels')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate of the optimizer')
parser.add_argument('--ckpt-dir', default='ckpt_resnet_proxy_nca_64/', type=Path, help='checkpoint folder to save model checkpoints')
parser.add_argument('--gpu', default=0, type=int, help='index of gpu to be used in training')

opt = parser.parse_args()

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([physical_devices[opt.gpu]], 'GPU')

imageroot = opt.dataroot / 'original/'

identities = np.load('identities.npy')
nclass = len(identities)

print(f'nclass: {nclass}')

print('Preparing dataset...')
dataset_cache_path = Path(f'siamese_data_cache_{opt.image_size}.npz')

if dataset_cache_path.is_file():
    cache = np.load(dataset_cache_path)
    imglist, ylist = cache['imglist'], cache['ylist']
else:
    imglist, ylist = [], []
    for i in range(nclass):
        _img_list = sorted(imageroot.glob(f'{identities[i]}/*.png'))
        imglist += [parse_image(str(p), [opt.image_size, opt.image_size]).numpy() for p in _img_list]
        ylist += [tf.one_hot(i, nclass).numpy() for _ in _img_list]
        print(f'Processed {len(imglist)} images...')
    imglist, ylist = np.array(imglist), np.array(ylist)
    shuffle_order = np.random.permutation(len(imglist))
    imglist, ylist = imglist[shuffle_order], ylist[shuffle_order]
    np.savez(dataset_cache_path, imglist=imglist, ylist=ylist)

print('Dataset has been fully loaded on memory.')


# Create the dataset
dataset = tf.data.Dataset.from_tensor_slices((imglist, ylist))\
    .shuffle(65536, reshuffle_each_iteration=True)\
    .batch(opt.batch_size)
dataset_val = tf.data.Dataset.from_tensor_slices((imglist[:7000], ylist[:7000]))\
    .batch(opt.batch_size)


class CustomEvaluation(Callback):
    def __init__(self, validation_data, interval=10):
        super(CustomEvaluation, self).__init__()
        self.validation_data = validation_data
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval != 0:
            return

        nclass = self.model.output_shape[-1]

        get_intermediate_layer_output = Model(inputs=self.model.inputs,
                                              outputs=self.model.get_layer('predictions').output)
        y_given = []
        y_embedding = []

        for x_val, y_val in self.validation_data:
            y_given.append(y_val)
            y_embedding_tmp = get_intermediate_layer_output(x_val)
            y_embedding.append(y_embedding_tmp)

        y_given = np.concatenate(y_given, axis=0)
        y_given_class_order = np.argsort(y_given, axis=-1)
        y_given_class = np.transpose(y_given_class_order)[-1]
        y_embedding = np.concatenate(y_embedding, axis=0)

        nmi = compute_normalized_mutual_information(
            y_given_class,
            cluster_by_kmeans(y_embedding, nclass)
        )
        print(f'NMI: {nmi * 100:.3f}')

        Y = assign_by_euclidean_at_k(y_embedding, y_given_class, 8)

        recall = []
        for k in [1, 2, 4, 8]:
            recall_at_k = compute_recall_at_k(y_given_class, Y, k)
            recall.append(recall_at_k)
            print(f'Recall@{k}: {100 * recall_at_k:.3f}')

        return nmi, recall


netS = ResnetProxyNCA(input_shape=(opt.image_size, opt.image_size, opt.nch), nclass=nclass)
netS.summary()

optimizer = Adam(learning_rate=opt.lr)
netS.compile(optimizer=optimizer, loss=nca_loss)

history = netS.fit(dataset, epochs=opt.epochs, callbacks=[
    CustomEvaluation(dataset_val, 50),
    ModelCheckpoint(opt.ckpt_dir/'ckpt-{epoch:03d}-{loss:.4f}', 'loss', save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau('loss', verbose=1)
], shuffle=False)
