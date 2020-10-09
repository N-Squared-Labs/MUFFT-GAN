import argparse

def load_args():
    arg_parser = argparse.ArgumentParser()

    # Loading, Inputs, Saving
    arg_parser.add_argument('--input_dir')                  # Directory of input image
    arg_parser.add_argument('--input_name')                 # Filename of input image
    arg_parser.add_argument('--translation')                # Type of Translation: Icy, Muddy, Wet, Grassy, Rocky
    arg_parser.add_argument('--img_channels', default=3)    # Number of channels of an input image
    arg_parser.add_argument('--output_dir')                 # Directory of output image

    # Network Hyper Parameters
    arg_parser.add_argument('--num_layers', default=5)      # Number of layers in each discriminator or generator
    arg_parser.add_argument('--fcd', default=32)            # Fully Connected Depth for Last Layer
    arg_parser.add_argument('--ndf', default=32)            # Network discriminator base depth scaling
    arg_parser.add_argument('--ngf', default=32)            # Network generator depth scaling
    arg_parser.add_argument('--kernel_dim', default=3)      
    arg_parser.add_argument('--padding', default=0)
    arg_parser.add_argument('--stride', default=1)

    # Pyramid Parameters
    arg_parser.add_argument('--scale', default=0.75)        # Scale Factor between the pyramid layers
    arg_parser.add_argument('--noise_weight', default=0.1)  # Weight of the noise added between layers


    # Tuning Hyperparameters
    arg_parser.add_argument('--num_epochs', default=2000)   # Number of epochs which we train each scale


    return arg_parser

