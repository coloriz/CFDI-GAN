import os
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from itertools import cycle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.utils import plot_model
import tensorflow_probability as tfp

from net.cfdigan import DiscriminatorBase, double_discriminator
from net.cfdigan import GeneratorBase, double_generator
from net.resnet_proxy_nca import ResnetProxyNCA
import vutils

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataroot', default='dataset/celeba/', type=Path, help='path to dataset')
parser.add_argument('--nid', type=int, help='number of identities at most')
parser.add_argument('--data-augmentation', action='store_true', help='perform data augmentation')
parser.add_argument('--nets-ckpt', type=Path, required=True, help='path to netD checkpoint')
parser.add_argument('--stage', default=1, type=int, help='stage to train')
parser.add_argument('--image-size', default=128, type=int, help='the height / width of the input image to network')
parser.add_argument('--nch', default=4, type=int, help='size of input channels')
parser.add_argument('--mixup', action='store_true', help='use mixup loss')
parser.add_argument('--gan-w', default=1.0, type=float, help='weight of adversarial loss')
parser.add_argument('--siam-w', default=0.5, type=float, help='weight of siamese loss')
parser.add_argument('--mask-output', action='store_true', help='mask the output of netG')
parser.add_argument('--lr-d', default=0.0001, type=float, help='learning rate for discriminator')
parser.add_argument('--lr-m', default=0.0001, type=float, help='learning rate for generator')
parser.add_argument('--beta1', default=0.0, type=float, help='beta1 for adam')
parser.add_argument('--beta2', default=0.99, type=float, help='beta2 for adam')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--outf', default='samples/', type=Path, help='folder to output images')
parser.add_argument('--log-dir', default='logs/', type=Path, help='log folder to save training progresses')
parser.add_argument('--ckpt-dir', default='ckpt_pgan/', type=Path, help='checkpoint folder to save model checkpoints')
parser.add_argument('--ckpt-interval', default=1000, type=int, help='interval of steps to save a checkpoint')
parser.add_argument('--sample-interval', default=1000, type=int, help='interval of steps to save generated samples')
parser.add_argument('--gpu', default=0, type=int, help='index of gpu to be used in training')

opt = parser.parse_args()

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([physical_devices[opt.gpu]], 'GPU')

imageroot = opt.dataroot / 'original/'
landmarkroot = opt.dataroot / 'landmark/'
maskroot = opt.dataroot / 'mask/'

identities = np.load('identities.npy')
if opt.nid is not None:
    identities = identities[:opt.nid]
nclass = len(identities)

print(f'nclass: {nclass}')


class DatasetSampler:
    def __init__(self):
        self.img_list = []
        self.landmark_list = []
        self.mask_list = []
        self.y_list = []
        self.label_to_range = []

        counter = 0
        for i in range(nclass):
            _img_list = sorted([p.relative_to(imageroot) for p in imageroot.glob(f'{identities[i]}/*.png')])
            self.img_list += [os.fspath(imageroot / p) for p in _img_list]
            self.landmark_list += [os.fspath(landmarkroot / p) for p in _img_list]
            self.mask_list += [os.fspath(maskroot / p) for p in _img_list]
            self.y_list += [i for _ in _img_list]
            self.label_to_range.append((counter, counter + len(_img_list)))
            counter += len(_img_list)
        print('DatasetSampler initialized.')

    def __iter__(self):
        for choose_randomly in cycle([True, False]):
            # Get half of the images randomly
            if choose_randomly:
                first_class = random.randrange(nclass)
                second_class = random.randrange(nclass)
            # Get the half of images from the same class
            else:
                first_class = second_class = random.randrange(nclass)
            i = random.randrange(self.label_to_range[first_class][0], self.label_to_range[first_class][1])
            j = random.randrange(self.label_to_range[second_class][0], self.label_to_range[second_class][1])

            yield (self.img_list[i], self.img_list[j]), \
                  (self.landmark_list[i], self.landmark_list[j]), \
                  (self.mask_list[i], self.mask_list[j]), \
                  self.y_list[j], tf.cast(first_class == second_class, tf.float32)


def parse_img(filename, image_size, crop_rnd, minmax=(-1, 1)):
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (minmax[1] - minmax[0]) * image + minmax[0]
    shape = tf.shape(image)
    h, w = float(shape[-3]), float(shape[-2])

    center_h = h / 2 + 0.1 * h
    center_w = w / 2
    min_sz, max_sz = w / 2, (w - center_w) * 1.5
    diff_sz, crop_sz = (max_sz - min_sz) / 2, min_sz / 2
    offset_height = int(center_h - crop_sz - diff_sz * crop_rnd[1])
    offset_width = int(center_w - crop_sz - diff_sz * crop_rnd[0])
    target_height = int(2 * crop_sz + diff_sz * (crop_rnd[1] + crop_rnd[3]))
    target_width = int(2 * crop_sz + diff_sz * (crop_rnd[0] + crop_rnd[2]))

    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
    image = tf.image.resize(image, image_size)
    return image


def mapper(image_size, augment):
    def func(img, landmark, mask, label, is_same):
        if augment:
            crop_rnd_1 = tf.random.uniform([4])
            crop_rnd_2 = tf.random.uniform([4])
        else:
            crop_rnd_1 = crop_rnd_2 = [1, 1, 1, 1]

        img_1 = parse_img(img[0], image_size, crop_rnd_1)
        img_2 = parse_img(img[1], image_size, crop_rnd_2)
        landmark_1 = parse_img(landmark[0], image_size, crop_rnd_1)
        landmark_2 = parse_img(landmark[1], image_size, crop_rnd_2)
        mask_1 = parse_img(mask[0], image_size, crop_rnd_1, (0, 1))
        mask_2 = parse_img(mask[1], image_size, crop_rnd_2, (0, 1))

        return (img_1, img_2), (landmark_1, landmark_2), (mask_1, mask_2), label, is_same
    return func


# Create the dataset
dataset = tf.data.Dataset.from_generator(
    DatasetSampler,
    ((tf.string, tf.string), (tf.string, tf.string), (tf.string, tf.string), tf.int32, tf.float32))


def build_pretrained_siamese():
    resnet_proxy_nca = ResnetProxyNCA(input_shape=(opt.image_size, opt.image_size, 3), include_nca=False)
    resnet_proxy_nca.load_weights(opt.nets_ckpt).expect_partial()

    embedding = resnet_proxy_nca.get_layer(name='predictions').output
    normalized = 2 * tf.math.l2_normalize(embedding, axis=-1)

    model = Model(inputs=resnet_proxy_nca.inputs,
                  outputs=normalized,
                  name='siamese')
    model.trainable = False

    return model


netS = build_pretrained_siamese()
plot_model(netS, f'netS_{opt.image_size}.png', show_shapes=True)


netG = GeneratorBase(opt.nch, nclass)
netD = DiscriminatorBase(opt.nch)

# Initialize loss functions
mae_loss = MeanAbsoluteError()
mse_loss = MeanSquaredError()


# Define contrastive loss function
def contrastive_loss(output1, output2, target, margin=2):
    distances = tf.math.reduce_euclidean_norm(output2 - output1, axis=-1)
    losses = target * distances + (1 - target) * tf.nn.relu(margin - distances)
    return tf.math.reduce_mean(losses)


# Setup Adam optimizers
optimizerD = Adam(learning_rate=opt.lr_d, beta_1=opt.beta1, beta_2=opt.beta2, epsilon=1e-8)
optimizerG = Adam(learning_rate=opt.lr_m, beta_1=opt.beta1, beta_2=opt.beta2, epsilon=1e-8)

# Setup a log directory
log_dir = str(opt.log_dir / datetime.now().strftime(f'pgan-{opt.stage}-%Y%m%d-%H%M%S'))
file_writer = tf.summary.create_file_writer(log_dir)


def smoothed_label(shape, minval, maxval, dtype=tf.float32):
    ta = tf.TensorArray(dtype, shape[0])
    for i in range(shape[0]):
        label = tf.random.uniform([], minval, maxval, dtype)
        ta = ta.write(i, tf.fill(shape[1:], label))
    return ta.stack()


dist = tfp.distributions.Beta(0.2, 0.2)


def train_discriminator(dataset_iter):
    img_faces, img_landmarks, img_masks, label, is_same = next(dataset_iter)
    # Calculate loss on the all-fake batch
    netG_input = tf.concat([img_faces[0] * img_masks[0], img_landmarks[0]], -1)
    fake = netG([netG_input, label], training=False)
    if opt.mask_output:
        fake = img_faces[0] * img_masks[0] + fake * (1 - img_masks[0])

    # 구분자의 입력은 [완성 얼굴 3채널, face[0]의 랜드마크 1채널]
    netD_fake_input = tf.concat([fake, img_landmarks[0]], -1)
    # Calculate loss on all-real batch
    netD_real_input = tf.concat([img_faces[1], img_landmarks[1]], -1)

    with tf.GradientTape() as tape:
        if opt.mixup:
            lam = dist.sample()
            mixup = lam * netD_real_input + (1 - lam) * netD_fake_input
            output_mixup = netD(mixup, training=True)
            errD = mse_loss(lam * tf.ones_like(output_mixup), output_mixup)
        else:
            netD_fake_output = netD(netD_fake_input, training=True)
            netD_real_output = netD(netD_real_input, training=True)
            errD_fake = mse_loss(tf.zeros_like(netD_fake_output), netD_fake_output)
            errD_real = mse_loss(tf.ones_like(netD_real_output) * 0.9, netD_real_output)
            errD = errD_fake + errD_real

    gradients = tape.gradient(errD, netD.trainable_variables)
    optimizerD.apply_gradients(zip(gradients, netD.trainable_variables))

    return errD


def train_generator(dataset_iter):
    img_faces, img_landmarks, img_masks, label, is_same = next(dataset_iter)
    netG_input = tf.concat([img_faces[0] * img_masks[0], img_landmarks[0]], -1)

    with tf.GradientTape() as tape:
        fake = netG([netG_input, label], training=True)
        if opt.mask_output:
            fake = img_faces[0] * img_masks[0] + fake * (1 - img_masks[0])

        netD_fake_input = tf.concat([fake, img_landmarks[0]], -1)
        netD_real_input = tf.concat([img_faces[1], img_landmarks[1]], -1)

        # Calculate GAN loss
        if opt.mixup:
            lam = dist.sample()
            mixup = lam * netD_real_input + (1 - lam) * netD_fake_input
            output_mixup = netD(mixup, training=False)
            errG_gan = mse_loss((1 - lam) * tf.ones_like(output_mixup), output_mixup)
        else:
            netD_fake_output = netD(netD_fake_input, training=False)
            errG_gan = mse_loss(tf.ones_like(netD_fake_output), netD_fake_output)
        # Calculate Siam loss
        fake = tf.image.resize(fake, [opt.image_size, opt.image_size], 'bicubic')
        real = tf.image.resize(img_faces[1], [opt.image_size, opt.image_size], 'bicubic')
        fc_fake = netS(fake, training=False)
        fc_real = netS(real, training=False)
        errG_siam = contrastive_loss(fc_fake, fc_real, tf.ones_like(is_same))
        errG = opt.gan_w * errG_gan + opt.siam_w * errG_siam

    gradients = tape.gradient(errG, netG.trainable_variables)
    optimizerG.apply_gradients(zip(gradients, netG.trainable_variables))

    return errG, errG_gan, errG_siam


@tf.function
def train_step(dataset_iter):
    """Define training step"""
    errD = train_discriminator(dataset_iter)
    errG, errG_gan, errG_siam = train_generator(dataset_iter)

    return errD, errG, errG_gan, errG_siam


stage = opt.stage

stage_to_resolution = {1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
minibatch_size = {1: 64, 2: 64, 3: 64, 4: 64, 5: 32, 6: 16}
tick_img = {1: 160, 2: 140, 3: 120, 4: 100, 5: 80, 6: 60}
# tick_img = {1: 160000, 2: 140000, 3: 120000, 4: 100000, 5: 80000, 6: 60000}

for _ in range(1, stage - 1):
    netD = double_discriminator(netD)
    netG = double_generator(netG)

if stage >= 2:
    # Load model weights from previous stage
    last_step = tick_img[stage - 1] // minibatch_size[stage - 1]
    netD.load_weights(opt.ckpt_dir/f'stage{stage - 1}'/f'netD-{last_step}')
    netG.load_weights(opt.ckpt_dir/f'stage{stage - 1}'/f'netG-{last_step}')
    # Double the model
    netD = double_discriminator(netD)
    netG = double_generator(netG)
    print(f'Stage {stage}: Models from stage {stage - 1} loaded.')
else:
    print('Initializing from scratch.')

plot_model(netD, f'netD_{stage}.png', show_shapes=True)
plot_model(netG, f'netG_{stage}.png', show_shapes=True)
batch_size = minibatch_size[stage]
sz = stage_to_resolution[stage]

stage_dataset = dataset.map(mapper([sz, sz], opt.data_augmentation), tf.data.experimental.AUTOTUNE, False) \
    .batch(batch_size) \
    .prefetch(tf.data.experimental.AUTOTUNE)
dataset_iter = iter(stage_dataset)
fixed_batch = next(iter(dataset.map(mapper([sz, sz], False)).batch(batch_size)))

# Check images from sampler
(img, _), (landmark, _), (mask, _), _, _ = next(dataset_iter)
img = (img + 1) / 2
landmark = tf.tile((landmark + 1) / 2, [1, 1, 1, 3])
mask = tf.tile(mask, [1, 1, 1, 3])
sample = tf.concat([img, landmark, mask], 2)
img_grid = vutils.make_grid(sample, nrow=4)
save_img(f'dataset_sample_{sz}.png', img_grid)

# Start training
nstep = tick_img[stage] // minibatch_size[stage]

for step in range(1, nstep + 1):
    # Set alpha
    alpha = min(step * 2 / nstep, 1.)
    if opt.stage >= 2:
        netD.get_layer('weighted_add').set_alpha(alpha)
        netG.get_layer('encoder_weighted_add').set_alpha(alpha)
        netG.get_layer('decoder_weighted_add').set_alpha(alpha)

    errD, errG, errG_gan, errG_siam = train_step(dataset_iter)
    # Output training stats
    if step % 1 == 0:
        print(f'stage {stage} [{step}/{nstep}]'.ljust(20) +
              f'alpha: {alpha:.4f}  '
              f'errD: {errD:.4f}  '
              f'errG: {errG:.4f}  '
              f'errG_gan: {errG_gan:.4f}  '
              f'errG_siam: {errG_siam:.4f}')

    # Log training stats
    with file_writer.as_default():
        tf.summary.scalar(f'stage{stage}_errD', errD, step=step)
        tf.summary.scalar(f'stage{stage}_errG', errG, step=step)
        tf.summary.scalar(f'stage{stage}_errG_gan', errG_gan, step=step)
        tf.summary.scalar(f'stage{stage}_errG_siam', errG_siam, step=step)

    # Save a model checkpoint
    if (step % opt.ckpt_interval == 0) or (step == nstep):
        netD.save_weights(opt.ckpt_dir/f'stage{stage}'/f'netD-{step}')
        netG.save_weights(opt.ckpt_dir/f'stage{stage}'/f'netG-{step}')
        print(f'Saved checkpoint at step {step}')

    # Check how the generator is doing by saving G's output on fixed batch
    if (step % opt.sample_interval == 0) or (step == nstep):
        img_faces, img_landmarks, img_masks, label, _ = fixed_batch
        netG_input = tf.concat([img_faces[0] * img_masks[0], img_landmarks[0]], -1)
        fake = netG([netG_input, label], training=False)
        if opt.mask_output:
            fake = img_faces[0] * img_masks[0] + fake * (1 - img_masks[0])
        img_triplet = tf.concat([img_faces[0], img_faces[1], fake], 2)
        img_grid = vutils.make_grid(img_triplet, nrow=4)
        img_grid = tf.clip_by_value(img_grid, -1, 1)
        img_grid = (img_grid + 1) / 2

        img_size_repr = f'{sz}x{sz}'
        with file_writer.as_default():
            tf.summary.image(f'Image triplets ({img_size_repr})', img_grid[tf.newaxis, ...], step=step)
        if not opt.outf.is_dir():
            opt.outf.mkdir(parents=True)
        save_img(opt.outf / f'stage_{stage}_img_triplet_{img_size_repr}_step_{step:07d}.png', img_grid)
