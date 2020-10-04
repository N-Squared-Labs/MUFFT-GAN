from config import load_args
from train_utils.train_pyramid import *
from train_utils.util_functions import *
from models.base_models import *

if __name__ == '__main__':
    # Load and Manipulate Config Parameters
    arg_parser = load_args()
    opt = arg_parser.parse_args
    opt = init_config(opt)
    
    # Create Training Variables and call Training
    real_img = read_input(opt)
    gens = []
    noise = []
    reals = []
    noise_amps = []
    train_pyramid(opt, gens, noise, reals, noise_amps)





