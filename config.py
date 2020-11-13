import argparse

def load_args():
    arg_parser = argparse.ArgumentParser()

    # Loading, Inputs, Saving
    arg_parser.add_argument('--input_dir_A', default='datasets/A')                  # Directory of input domain A
    arg_parser.add_argument('--input_dir_B', default='datasets/B')                  # Directory of input domain B
    arg_parser.add_argument('--translation')                # Type of Translation: Icy, Muddy, Wet, Grassy, Rocky
    arg_parser.add_argument('--img_size', default=256, type=int)       # Size that input images get resized to
    arg_parser.add_argument('--img_channels', default=3, type=int)    # Number of channels of an input image
    arg_parser.add_argument('--output_dir')                 # Directory of output image
    arg_parser.add_argument('--batch_size', default=8, type=int)      # Batch size for dataset
    arg_parser.add_argument('--device', default='cuda:0')
    arg_parser.add_argument('--snapshot_interval', default=50)        # Epochs per snapshot network and produce image grid 

    # Network Hyper Parameters
    arg_parser.add_argument('--num_layers', default=5, type=int)      # Number of layers in each discriminator or generator
    arg_parser.add_argument('--num_nce_layers', default='0,4,8,12,16', type=str)
    arg_parser.add_argument('--resnet_blocks', default=6, type=int)
    arg_parser.add_argument('--fcd', default=32, type=int)            # Fully Connected Depth for Last Layer
    arg_parser.add_argument('--nz', default=100, type=int)
    arg_parser.add_argument('--ndf', default=32, type=int)            # Network discriminator base depth scaling
    arg_parser.add_argument('--ngf', default=32, type=int)            # Network generator depth scaling
    arg_parser.add_argument('--kernel_dim', default=4, type=int)      
    arg_parser.add_argument('--padding', default=1, type=int)
    arg_parser.add_argument('--stride', default=2, type=int)
    arg_parser.add_argument('--lr', default=0.0002, type=float)

    # Pyramid Parameters
    arg_parser.add_argument('--scale', default=0.75, type=float)        # Scale Factor between the pyramid layers
    arg_parser.add_argument('--noise_weight', default=0.1, type=float)  # Weight of the noise added between layers


    # Tuning Hyperparameters
    arg_parser.add_argument('--num_epochs', default=100, type=int)   # Number of epochs which we train each scale
    arg_parser.add_argument('--lambda_G', default=1, type=float)


    return arg_parser

