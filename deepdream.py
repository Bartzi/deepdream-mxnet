import json

import argparse
import mxnet as mx

from collections import OrderedDict

import os

import sys
from mxnet.visualization import print_summary
from tqdm import tqdm

from visualization import LaplaceFeatureVisualization, TiledFeatureVisualization


def get_all_layernames(nodes):
    layers = OrderedDict()
    for i, node in enumerate(nodes):
        if node['op'] != 'null':
            layers[node['name']] = i
    return layers


def rework_symbol_file(prefix, layername):
    with open("orig-{}-symbol.json".format(prefix)) as json_file:
        symbol_json = json.load(json_file)
    layers = get_all_layernames(symbol_json['nodes'])

    # strip all unecessary parts of network and add new loss layers
    head_id = layers[layername]
    symbol_json['nodes'] = symbol_json['nodes'][:head_id + 1]
    symbol_json['arg_nodes'] = list(filter(lambda x: x <= head_id, symbol_json['arg_nodes']))
    symbol_json['heads'] = [[len(symbol_json['nodes']) - 1, 0]]

    num_filters = int(symbol_json['nodes'][head_id]['param']['num_filter'])

    with open('{}-symbol.json'.format(prefix), 'w') as json_file:
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
    parser.add_argument('layer_name', help='name of layer to visualize')
    parser.add_argument('layer_id', type=int, help='number of channel to visualize')
    parser.add_argument('-g', '--gpu', type=int, default=-1, help='gpu to use [default: cpu]')
    parser.add_argument('--all', action='store_true', default=False, help='visualize all channels of given layer')
    parser.add_argument('--folder', help='folder where images shall be saved, if no folder is given it will be shown to the user')

    args = parser.parse_args()

    if args.all and not args.folder:
        print("Please do not use the switch all without the switch folder!")
        sys.exit(1)

    # adjust symbol file and get number of filters in chosen layer
    number_of_filters = rework_symbol_file(args.model_prefix, args.layer_name)

    if args.all:
        filters_to_visualize = list(range(number_of_filters))
    else:
        filters_to_visualize = [args.layer_id]

    context = mx.cpu() if args.gpu < 0 else mx.gpu(args.gpu)

    # build model
    symbol, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, 0000)
    print_summary(symbol, shape={'data': (1, 3, 224, 224)})

    # prepare input data
    data = mx.nd.random.uniform(shape=(1, 3, 224, 224), ctx=context)

    # do the visualization
    for filter_id in tqdm(filters_to_visualize):
        vis_symbol = add_vis_layers(symbol, filter_id)

        mod = mx.mod.Module(vis_symbol, context=mx.gpu(0))
        mod.bind(data_shapes=[('data', (1, 3, 224, 224))], inputs_need_grad=True)
        mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True, allow_extra=True)

        # preform visualization
        laplace_visualizer = LaplaceFeatureVisualization(mod, context)
        image = laplace_visualizer.visualize(data.copy())
        if args.folder:
            os.makedirs(args.folder, exist_ok=True)
            image.save(os.path.join(args.folder, "{}.png".format(filter_id)))
        else:
            image.show()
