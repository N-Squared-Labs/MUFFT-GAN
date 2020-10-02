from config import load_args
from train_utils.train_pyramid import *
from train_utils.util_functions import *
from models.base_models import *

if __name__ == '__main__':
    # Load and Manipulate Config Parameters
    arg_parser = load_args()
    arg_parser.add_argument('--input_dir', help='Directory of input image')
    arg_parser.add_argument('--input_name', help='Filename of input image')
    arg_parser.add_argument('--translation', help='Type of Translation: Icy, Muddy, Wet, Grassy, Rocky')
    opt = arg_parser.parse_args
    opt = init_config(opt)

    # Generate and Check Output Directory
    output_dir = generate_output_dir(opt)
    if(os.path.exists(output_dir)):
        print('Model has already been generated')
        exit(1)
    else:
        os.makedirs(output_dir)
    
    # Create Training Variables and call Training
    




