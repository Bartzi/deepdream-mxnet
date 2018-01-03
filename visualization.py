from collections import namedtuple

import mxnet as mx
import numpy as np
from PIL import Image

from tqdm import tqdm


Batch = namedtuple('Batch', ['data'])


def normalize_array(array):
    min = array.min()
    array -= min
    max = array.max()
    array /= max
    return array


def array_to_image(array):
    normalized_array = normalize_array(array) * 255
    normalized_array = normalized_array.astype('uint8').transpose(1, 2, 0)
    image = Image.fromarray(normalized_array)
    return image


class FeatureVisualization:

    def __init__(self, module, context):
        self.module = module
        self.context = context

    def visualize(self, data):
        image = None
        for _ in tqdm(range(40)):
            self.module.forward(Batch([data]))
            self.module.backward()

            input_gradients = self.module.get_input_grads()[0].asnumpy()
            input_gradients /= input_gradients.std() + 1e-8

            data += mx.nd.array(input_gradients, ctx=self.context)
            image = array_to_image(data.asnumpy()[0])
        return image


class TiledFeatureVisualization(FeatureVisualization):

    def calc_grad_tiled(self, data, tile_size=224):
        image_height, image_width = data.shape[-2:]
        shift_x, shift_y = np.random.randint(tile_size, size=2)

        shifted_image = np.roll(np.roll(data.asnumpy(), shift_x, 3), shift_y, 2)
        grad = mx.nd.zeros_like(data, ctx=self.context)

        for y in range(0, max(image_height - tile_size // 2, tile_size), tile_size):
            for x in range(0, max(image_width - tile_size // 2, tile_size), tile_size):
                tiled_crop = mx.nd.array(shifted_image[:, :, y:y+tile_size, x:x+tile_size], ctx=self.context)
                self.module.forward_backward(Batch([tiled_crop]))
                gradients = self.module.get_input_grads()[0]
                grad[:, :, y:y+tile_size, x:x+tile_size] = gradients
        return np.roll(np.roll(grad.asnumpy(), -shift_x, 3), -shift_y, 2)

    def resize_data(self, data, scale):
        zoom_matrix = mx.nd.array([[1, 0, 0], [0, 1, 0]], ctx=self.context)
        zoom_matrix = mx.nd.reshape(zoom_matrix, shape=(1, 6))

        _, _, height, width = data.shape
        new_height = int(height * scale)
        new_width = int(width * scale)

        grid = mx.nd.GridGenerator(data=zoom_matrix, transform_type='affine', target_shape=(new_height, new_width))
        data = mx.nd.BilinearSampler(data, grid)

        return data

    def visualize(self, data):
        image = None
        num_octaves = 3
        num_steps = 10
        octave_scale = 1.5

        for octave in tqdm(range(num_octaves)):
            if octave > 0:
                data = self.resize_data(data, octave_scale)

        for i in range(num_steps):
            g = self.calc_grad_tiled(data)
            g /= g.std() + 1e-8
            data += mx.nd.array(g, ctx=self.context)
            image = array_to_image(data.asnumpy()[0])
        return image


class LaplaceFeatureVisualization(TiledFeatureVisualization):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        laplace_kernel = np.float32([1, 4, 6, 4, 1])
        laplace_kernel = np.outer(laplace_kernel, laplace_kernel)
        laplace_kernel_5x5 = laplace_kernel[:, :, None, None] / laplace_kernel.sum() * np.eye(3, dtype=np.float32)
        self.laplace_kernel_5x5 = mx.nd.array(laplace_kernel_5x5.transpose(2, 3, 0, 1), ctx=self.context)

    def laplace_split(self, data):
        low = mx.nd.Convolution(
            data,
            self.laplace_kernel_5x5,
            kernel=(5, 5),
            stride=(2, 2),
            no_bias=True,
            num_filter=3
        )
        if data.shape[-2] % 2 == 0:
            low_2 = mx.nd.Deconvolution(
                low,
                self.laplace_kernel_5x5 * 4,
                kernel=(5, 5),
                stride=(2, 2),
                num_filter=3,
                no_bias=True,
                adj=(1, 1)
            )
        else:
            low_2 = mx.nd.Deconvolution(
                low,
                self.laplace_kernel_5x5 * 4,
                kernel=(5, 5),
                stride=(2, 2),
                num_filter=3,
                no_bias=True
            )
        high = data - low_2
        return low, high

    def laplace_split_n(self, data, n):
        levels = []
        for i in range(n):
            data, high = self.laplace_split(data)
            levels.append(high)
        levels.append(data)
        return levels[::-1]

    def laplace_merge(self, levels):
        data = levels[0]
        for high in levels[1:]:
            if high.shape[-2] % 2 == 0:
                data = mx.nd.Deconvolution(
                    data,
                    self.laplace_kernel_5x5 * 4,
                    kernel=(5, 5),
                    stride=(2, 2),
                    num_filter=3,
                    no_bias=True,
                    adj=(1, 1)
                )
            else:
                data = mx.nd.Deconvolution(
                    data,
                    self.laplace_kernel_5x5 * 4,
                    kernel=(5, 5),
                    stride=(2, 2),
                    num_filter=3,
                    no_bias=True
                )
            data += high

        return data

    def normalize_std(self, data, eps=1e-10):
        std = mx.nd.sqrt(mx.nd.mean(mx.nd.square(data)))
        return data / mx.nd.maximum(std, eps)

    def laplacian_normalization(self, data, scale_n=4):
        tlevels = self.laplace_split_n(data, scale_n)
        tlevels = list(map(self.normalize_std, tlevels))
        out = self.laplace_merge(tlevels)
        return out

    def visualize(self, data):
        image = None
        num_octaves = 3
        num_steps = 20
        octave_scale = 1.5

        for octave in tqdm(range(num_octaves)):
            if octave > 0:
                data = self.resize_data(data, octave_scale)

            for i in range(num_steps):
                g = self.calc_grad_tiled(data)
                g = self.laplacian_normalization(mx.nd.array(g, ctx=self.context))
                data += g
                image = array_to_image(data.asnumpy()[0])
        return image
