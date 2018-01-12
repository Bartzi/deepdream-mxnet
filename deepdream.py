import json

import argparse
import mxnet as mx

from collections import OrderedDict

import os

import sys

import numpy as np
from tqdm import tqdm

from visualization import LaplaceFeatureVisualization


def get_all_layernames(nodes):
    layers = OrderedDict()
    for i, node in enumerate(nodes):
        if node['op'] != 'null':
            layers[node['name']] = i
    return layers


def rework_symbol_file(prefix, layername):
    working_dir = os.path.dirname(prefix)
    prefix = os.path.basename(prefix)

    with open(os.path.join(working_dir, "orig-{}-symbol.json".format(prefix))) as json_file:
        symbol_json = json.load(json_file)
    layers = get_all_layernames(symbol_json['nodes'])

    # strip all unecessary parts of network and add new loss layers
    head_id = layers[layername]
    symbol_json['nodes'] = symbol_json['nodes'][:head_id + 1]
    symbol_json['arg_nodes'] = list(filter(lambda x: x <= head_id, symbol_json['arg_nodes']))
    symbol_json['heads'] = [[len(symbol_json['nodes']) - 1, 0]]

    try:
        num_filters = int(symbol_json['nodes'][head_id]['param']['num_filter'])
    except KeyError:
        try:
            num_filters = int(symbol_json['nodes'][head_id]['attr']['num_filter'])
        except KeyError:
            num_filters = int(symbol_json['nodes'][head_id]['attr']['num_hidden'])

    with open(os.path.join(working_dir, '{}-symbol.json'.format(prefix)), 'w') as json_file:
        json.dump(symbol_json, json_file, indent=4)

    return num_filters


def add_vis_layers(symbol, channel):
    sliced = mx.sym.slice_axis(symbol, axis=1, begin=channel, end=channel+1, name='slice_axis')
    mean = mx.sym.mean(sliced, axis=0, exclude=True, keepdims=True, name='mean')
    loss = mx.sym.MakeLoss(mean, name='makeloss')

    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tool that does deep dream like vis of feature maps')
    parser.add_argument('model_prefix', help='model prefix')
    parser.add_argument('epoch', help='epoch to load', type=int)
    parser.add_argument('layer_name', help='name of layer to visualize')
    parser.add_argument('layer_id', type=int, help='number of channel to visualize')
    parser.add_argument('-g', '--gpu', type=int, default=-1, help='gpu to use [default: cpu]')
    parser.add_argument('--all', action='store_true', default=False, help='visualize all channels of given layer')
    parser.add_argument('--folder', help='folder where images shall be saved, if no folder is given it will be shown to the user')
    parser.add_argument('-c', '--config', default='config.json', help='path to config file (json format) [default: config.json]')

    args = parser.parse_args()

    if args.all and not args.folder:
        print("Please do not use the switch `all` without the switch `folder`!")
        sys.exit(1)

    # adjust symbol file and get number of filters in chosen layer
    number_of_filters = rework_symbol_file(args.model_prefix, args.layer_name)

    if args.all:
        filters_to_visualize = list(range(number_of_filters))
    else:
        filters_to_visualize = [args.layer_id]

    with open(args.config) as config_file:
        config = json.load(config_file)

    context = mx.cpu() if args.gpu < 0 else mx.gpu(args.gpu)

    # build model
    data_shape = config['input_shape']
    batch_size = config.get('batch_size', 1)
    data_shape = [batch_size] + data_shape
    symbol, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, args.epoch)

    # prepare input data
    fixed_generator = np.random.RandomState(42)
    data = fixed_generator.uniform(size=data_shape)
    mean = config.get('mean', None)
    if mean is not None:
        assert data_shape[1] == len(mean), "the mean to subtract does not have the correct amount of values"
        data *= 255
        for i, mean_value in enumerate(mean):
            data[:, i, ...] -= mean_value
    data = mx.nd.array(data, ctx=context)

    # do the visualization
    for filter_id in tqdm(filters_to_visualize):
        vis_symbol = add_vis_layers(symbol, filter_id)

        mod = mx.mod.Module(vis_symbol, context=context)
        mod.bind(data_shapes=[('data', data_shape)], inputs_need_grad=True)
        mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True, allow_extra=True)

        # preform visualization
        laplace_visualizer = LaplaceFeatureVisualization(
            mod,
            context,
            scale_n=config['scale_n'],
            num_steps=config['num_steps'],
            num_octaves=config['num_octaves'],
            octave_scale=config['octave_scale'],
            max_tile_size=config['max_tile_size'],
            step_size=config['step_size'],
            data_shape=data_shape,
        )

        images = laplace_visualizer.visualize(data.copy())
        if args.folder:
            os.makedirs(args.folder, exist_ok=True)
            for i, image in enumerate(images):
                image.save(os.path.join(args.folder, "{}_{}.png".format(filter_id, i)))
        else:
            images[0].show()
