import argparse


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


def get_args(txt_file=None):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # checkpoint (only needed when testing the model)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--encoder_path', type=str, default=None)

    # ↓↓↓↓
    # NOTE: project-specific args
    parser.add_argument('--output_dim', type=int, default=3, help='{3, 4}')
    parser.add_argument('--output_type', type=str, default='R', help='{R, G}')
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)

    parser.add_argument('--encoder_B', type=int, default=5)

    parser.add_argument('--decoder_NF', type=int, default=2048)
    parser.add_argument('--decoder_BN', default=False, action="store_true")
    parser.add_argument('--decoder_down', type=int, default=2)
    parser.add_argument('--learned_upsampling', default=False, action="store_true")

    # read arguments from txt file
    if txt_file:
        config_filename = '@' + txt_file

    args = parser.parse_args([config_filename])
    return args
