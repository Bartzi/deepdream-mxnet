# deepdream-mxnet
An implementation of deepdream for mxnet

# Installation

0. **Recommended** Create a new [Virtualenvironment](https://virtualenv.pypa.io/en/stable/)
1. Install MXNet in your venv (compile it by yourself, or use pip (`pip install mxnet`)
2. Install all other requirements with `pip install -r requirements.txt`
3. Profit

# Usage

Assume you are having the following files:
- Inception-BN-0126.params (trained model)
- Inception-BN-symbol.json (network definition)

1. You will need to rename `Inception-BN-symbol.json` to `orig-Inception-BN-symbol.json`
2. Choose the layer you want to visualize by looking at the names of the `Convolutional`
layers in the symbol definition file
3. Remember how many channels this layer has.
4. Set all required values in the config file (`config.json`)
    1. `input_shape` the shape of the input images in the form `num_channels, height, width`
    2. `batch_size` the batch size to use while dreaming
    3. `scale_n` the number of downscale steps for the laplacian gradient normalization (2^scale_n should be smaller than your image size)
    4. `num_steps` number of optimization steps per octave
    5. `num_octaves` number of times the image shall be increased in size
    6. `octave_scale` how much the size should increase
    7. `step_size` step size for applying the gradient on the input iamge
    8. `max_tile_size` max size of each tile in pixels for saving GPU memory
    9. `mean` RGB mean values that should be subtracted from the input image (not mandatory)
5. Start the visualization with: `python deepdream.py <model-prefix> <epoch> <layer_name> <layer_id> -g <gpu_to_use> --all -c config.json --folder <place to save resulting images>`,
with our example data: `python deepdream.py Inception-BN 0126 conv_4d_double_3x3_1 140 --all -c config.json -g 0 --folder images/inception`
6. Profit again!

# Questions?

Feel free to open an issue.

# Improvements?

I'm happy to review your Pull Request!
