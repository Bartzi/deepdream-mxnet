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
    if normalized_array.shape[-1] == 1:
        normalized_array = np.broadcast_to(normalized_array, normalized_array.shape[:-1] + (3, ))
    image = Image.fromarray(normalized_array)
    return image


class FeatureVisualization:

    def __init__(self, module, context, step_size=1):
        self.module = module
        self.context = context
        self.step_size = step_size

    def visualize(self, data):
        image = None
        for _ in tqdm(range(40)):
            self.module.forward(Batch([data]))
            self.module.backward()

            input_gradients = self.module.get_input_grads()[0].asnumpy()
            input_gradients /= input_gradients.std() + 1e-8

            data += self.step_size * mx.nd.array(input_gradients, ctx=self.context)
            image = array_to_image(data.asnumpy()[0])
        return image


class TiledFeatureVisualization(FeatureVisualization):

    def __init__(self, *args, **kwargs):
        self.max_tile_size = kwargs.pop('max_tile_size', 256)
        self.num_octaves = kwargs.pop('num_octaves', 3)
        self.num_steps = kwargs.pop('num_steps', 10)
        self.octave_scale = kwargs.pop('octave_scale', 1.5)
        super().__init__(*args, **kwargs)

    def calc_grad_tiled(self, data):
        image_height, image_width = data.shape[-2:]
        tile_size = self.max_tile_size
        shift_x, shift_y = np.random.randint(tile_size, size=2)

        shifted_image = np.roll(np.roll(data.asnumpy(), shift_x, 3), shift_y, 2)
        grad = mx.nd.zeros_like(data, ctx=self.context)

        for y in range(0, max(image_height - tile_size, tile_size), tile_size):
            for x in range(0, max(image_width - tile_size, tile_size), tile_size):
                y_end = min(y + tile_size, image_height)
                x_end = min(x + tile_size, image_width)
                tiled_crop = mx.nd.array(shifted_image[:, :, y:y_end, x:x_end], ctx=self.context)
                self.module.forward(Batch([tiled_crop]))
                self.module.backward()
                gradients = self.module.get_input_grads()[0]
                grad[:, :, y:y_end, x:x_end] = gradients
        return np.roll(np.roll(grad.asnumpy(), -shift_x, 3), -shift_y, 2)

    def resize_data(self, data, scale):
        batch_size, _, height, width = data.shape
        zoom_matrix = mx.nd.array([[1, 0, 0], [0, 1, 0]], ctx=self.context)
        zoom_matrix = mx.nd.reshape(zoom_matrix, shape=(1, 6))
        zoom_matrix = mx.nd.broadcast_axis(zoom_matrix, 0, batch_size)

        new_height = int(height * scale)
        new_width = int(width * scale)

        grid = mx.nd.GridGenerator(data=zoom_matrix, transform_type='affine', target_shape=(new_height, new_width))
        data = mx.nd.BilinearSampler(data, grid)

        return data

    def visualize(self, data):
        image = None

        for octave in tqdm(range(self.num_octaves)):
            if octave > 0:
                data = self.resize_data(data, self.octave_scale)

        for i in range(self.num_steps):
            g = self.calc_grad_tiled(data)
            g /= g.std() + 1e-8
            data += self.step_size * mx.nd.array(g, ctx=self.context)
            image = array_to_image(data.asnumpy()[0])
        return image


class LaplaceFeatureVisualization(TiledFeatureVisualization):

    def __init__(self, *args, **kwargs):
        data_shape = kwargs.pop('data_shape')
        self.num_channels = data_shape[1]
        self.scale_n = kwargs.pop('scale_n', 4)
        super().__init__(*args, **kwargs)

        laplace_kernel = np.float32([1, 4, 6, 4, 1])
        laplace_kernel = np.outer(laplace_kernel, laplace_kernel)
        laplace_kernel_5x5 = laplace_kernel[:, :, None, None] / laplace_kernel.sum() * np.eye(self.num_channels, dtype=np.float32)
        self.laplace_kernel_5x5 = mx.nd.array(laplace_kernel_5x5.transpose(2, 3, 0, 1), ctx=self.context)

    def laplace_split(self, data):
        low = mx.nd.Convolution(
            data,
            self.laplace_kernel_5x5,
            kernel=(5, 5),
            stride=(2, 2),
            no_bias=True,
            num_filter=self.num_channels,
        )
        if data.shape[-2] % 2 == 0:
            low_2 = mx.nd.Deconvolution(
                low,
                self.laplace_kernel_5x5 * 4,
                kernel=(5, 5),
                stride=(2, 2),
                num_filter=self.num_channels,
                no_bias=True,
                adj=(1, 1)
            )
        else:
            low_2 = mx.nd.Deconvolution(
                low,
                self.laplace_kernel_5x5 * 4,
                kernel=(5, 5),
                stride=(2, 2),
                num_filter=self.num_channels,
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
                    num_filter=self.num_channels,
                    no_bias=True,
                    adj=(1, 1)
                )
            else:
                data = mx.nd.Deconvolution(
                    data,
                    self.laplace_kernel_5x5 * 4,
                    kernel=(5, 5),
                    stride=(2, 2),
                    num_filter=self.num_channels,
                    no_bias=True
                )
            data += high

        return data

    def normalize_std(self, data, eps=1e-10):
        std = mx.nd.sqrt(mx.nd.mean(mx.nd.square(data)))
        return data / mx.nd.maximum(std, eps)

    def laplacian_normalization(self, data):
        tlevels = self.laplace_split_n(data, self.scale_n)
        tlevels = list(map(self.normalize_std, tlevels))
        out = self.laplace_merge(tlevels)
        return out

    def visualize(self, data):
        images = None

        for octave in tqdm(range(self.num_octaves)):
            if octave > 0:
                data = self.resize_data(data, self.octave_scale)

            for i in range(self.num_steps):
                g = self.calc_grad_tiled(data)
                g = self.laplacian_normalization(mx.nd.array(g, ctx=self.context))
                data += self.step_size * g
                array_to_image(data.asnumpy()[0])
        individual_images = mx.nd.split(data, len(data), 0, squeeze_axis=True)
        if not hasattr(individual_images, '__iter__'):
            individual_images = [individual_images]
        images = [array_to_image(i.asnumpy()) for i in individual_images]
        return images
