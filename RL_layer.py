import tensorflow as tf
import numpy as np


class RL_Deconv(tf.keras.layers.Layer):
    def __init__(self, kernel=3, sigma=1.0, channels=3,
                 iterations=1000, tolerance=1e-5):
        super(RL_Deconv, self).__init__()
        self.channels = channels
        self.iterations = iterations
        self.tolerance = tolerance
        self.pad_size = kernel // 2
        # TODO set up kernel from outside
        kernel_tmp = np.zeros((kernel, kernel, channels, 1), dtype=np.float32)
        divisor = -2.0 * sigma * sigma
        norm_coef = 0
        for i in range(-self.pad_size, self.pad_size+1):
            for j in range(-self.pad_size, self.pad_size+1):
                value = np.exp((i * i + j * j) / divisor)
                kernel_tmp[i + self.pad_size, j + self.pad_size, :, 0] = value
                norm_coef += value

        kernel_tmp = kernel_tmp / norm_coef

        self.kernel = tf.constant(kernel_tmp,
                                  dtype=tf.float32)
        self.kernel_flipped = tf.constant(np.flip(kernel_tmp, axis=(0, 1)),
                                          dtype=tf.float32)

    def call(self, inputs):
        latent_est = inputs
        paddings = ((0, 0), (self.pad_size, self.pad_size),
                    (self.pad_size, self.pad_size), (0, 0))
        for i in range(self.iterations):
            latent_est_pad = tf.pad(
                latent_est, paddings, mode='REFLECT')
            est_conv = tf.nn.depthwise_conv2d(
                latent_est_pad,
                self.kernel,
                strides=(1, 1, 1, 1),
                padding='VALID')
            relative_blur = inputs / est_conv
            relative_blur_pad = tf.pad(
                relative_blur, paddings, mode='REFLECT')
            error_est = tf.nn.depthwise_conv2d(
                relative_blur_pad,
                self.kernel_flipped,
                strides=(1, 1, 1, 1),
                padding='VALID')
            latent_est = latent_est * error_est

            if tf.abs(1 - tf.reduce_mean(error_est)) < self.tolerance:
                print(i)
                break

        return latent_est


layer = RL_Deconv()

from PIL import Image, ImageFilter
import cv2

img = Image.open('Lenna_(test_image).png')
img = np.asarray(img, dtype=np.float32)
img = cv2.GaussianBlur(img, (3, 3), 1.0)
Image.fromarray(np.uint8(img)).save('blurred.png')

img = np.asarray(img, dtype=np.float32)
img = np.expand_dims(img, axis=0)
res = layer(img)
res = np.squeeze(res)
Image.fromarray(np.uint8(res)).save('res.png')
