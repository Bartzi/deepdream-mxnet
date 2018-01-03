import json
import mxnet as mx

from collections import namedtuple, OrderedDict

import numpy as np
import tqdm as tqdm
from PIL import Image
from mxnet.visualization import print_summary

Batch = namedtuple('Batch', ['data'])


def get_all_layernames(nodes):
    layers = OrderedDict()
    for i, node in enumerate(nodes):
        if node['op'] != 'null':
            layers[node['name']] = i
    return layers


def rework_symbol_file(prefix, layername, channel):
    with open("orig-{}-symbol.json".format(prefix)) as json_file:
        symbol_json = json.load(json_file)
    layers = get_all_layernames(symbol_json['nodes'])

    # strip all unecessary parts of network and add new loss layers
    head_id = layers[layername]
    symbol_json['nodes'] = symbol_json['nodes'][:head_id + 1]

    symbol_json['nodes'].append({
        "op": "slice_axis",
        "name": "slice_axis",
        "attrs": {
            "axis": "1",
            "begin": str(channel),
            "end": str(channel + 1),
        },
        "inputs": [[len(symbol_json['nodes']) - 1, 0, 0]],
    })

    symbol_json['nodes'].append({
        'op': 'mean',
        'name': 'mean',
        "attrs": {
            "axis": "0",
            "exclude": "True",
            "keepdims": "True",
        },
        'inputs': [[len(symbol_json['nodes']) - 1, 0, 0]]
    })

    symbol_json['nodes'].append({
        'op': 'MakeLoss',
        'name': 'makeloss',
        'inputs': [[len(symbol_json['nodes']) - 1, 0, 0]]
    })

    symbol_json['arg_nodes'] = list(filter(lambda x: x <= head_id, symbol_json['arg_nodes']))
    symbol_json['heads'] = [[len(symbol_json['nodes']) - 1, 0]]

    with open('{}-symbol.json'.format(prefix), 'w') as json_file:
        json.dump(symbol_json, json_file, indent=4)


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


def naive_feature_vis(mod, data, image):
    for _ in tqdm.tqdm(range(40)):
        mod.forward(Batch([data]))
        mod.backward()

        input_gradients = mod.get_input_grads()[0].asnumpy()
        input_gradients /= input_gradients.std() + 1e-8

        data += mx.nd.array(input_gradients, ctx=mx.gpu(0))
        image = array_to_image(data.asnumpy()[0])
    return image


def calc_grad_tiled(mod, data, tile_size=224):
    image_height, image_width = data.shape[-2:]
    shift_x, shift_y = np.random.randint(tile_size, size=2)

    shifted_image = np.roll(np.roll(data.asnumpy(), shift_x, 3), shift_y, 2)
    grad = mx.nd.zeros_like(data, ctx=mx.gpu(0))

    for y in range(0, max(image_height - tile_size // 2, tile_size), tile_size):
        for x in range(0, max(image_width - tile_size // 2, tile_size), tile_size):
            tiled_crop = mx.nd.array(shifted_image[:, :, y:y+tile_size, x:x+tile_size], ctx=mx.gpu(0))
            mod.forward_backward(Batch([tiled_crop]))
            gradients = mod.get_input_grads()[0]
            grad[:, :, y:y+tile_size, x:x+tile_size] = gradients
    return np.roll(np.roll(grad.asnumpy(), -shift_x, 3), -shift_y, 2)


def render_multiscale(mod, data, image):
    num_octaves = 5
    num_steps = 10
    octave_scale = 1.5
    zoom_matrix = mx.nd.array([[1, 0, 0], [0, 1, 0]], ctx=mx.gpu(0))
    zoom_matrix = mx.nd.reshape(zoom_matrix, shape=(1, 6))

    for octave in tqdm.tqdm(range(num_octaves)):
        if octave > 0:
            new_height = int(image.height * octave_scale)
            new_width = int(image.width * octave_scale)
            image = image.resize((new_width, new_height), Image.BILINEAR)

            grid = mx.nd.GridGenerator(data=zoom_matrix, transform_type='affine', target_shape=(new_height, new_width))
            data = mx.nd.BilinearSampler(data, grid)

        for i in range(num_steps):
            g = calc_grad_tiled(mod, data)
            g /= g.std() + 1e-8
            data += mx.nd.array(g, ctx=mx.gpu(0))
            image = array_to_image(data.asnumpy()[0])
    return image


if __name__ == "__main__":
    # adjust symbol file
    rework_symbol_file('caffenet', 'conv4', 100)

    # build model
    symbol, arg_params, aux_params = mx.model.load_checkpoint('caffenet', 0000)
    print_summary(symbol, shape={'data': (1, 3, 224, 224)})

    mod = mx.mod.Module(symbol, context=mx.gpu(0))
    mod.bind(data_shapes=[('data', (1, 3, 224, 224))], inputs_need_grad=True)
    mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True, allow_extra=True)

    # prepare input data
    data = mx.nd.random.uniform(shape=(1, 3, 224, 224), ctx=mx.gpu(0))

    image = array_to_image(data.asnumpy()[0])
    image = render_multiscale(mod, data, image)
    image.show()
