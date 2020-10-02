import argparse

def load_args():
    arg_parser = argparse.ArgumentParser()

    # Loading, Inputs, Saving
    arg_parser.add_argument('--')

    # Network Hyper Parameters
    arg_parser.add_argument('--img_channels', default=3)
    arg_parser.add_argument('--num_layers', default=5)
    arg_parser.add_argument('--fcd', default=32)        # Fully Connected Depth for Last Layer
    arg_parser.add_argument('--min_fcd', default=32)    # Minimum Fully Connected Depth for Last Layer
    arg_parser.add_argument('--kernel_dim', default=3)
    arg_parser.add_argument('--padding', default=0)
    arg_parser.add_argument('--stride', default=1)

    # Pyramid Parameters
    return arg_parser

