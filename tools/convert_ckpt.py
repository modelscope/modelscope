# Copyright (c) Alibaba, Inc. and its affiliates.
import shutil
import argparse
import torch
import os


def convert_single_pth(fullname):
    filename, ext = os.path.splitext(fullname)
    checkpoint = torch.load(fullname, map_location='cpu')
    only_module = 'state_dict' not in checkpoint
    state_dict = checkpoint if only_module else checkpoint['state_dict']
    torch.save(state_dict, fullname)

    if not only_module:
        checkpoint.pop('state_dict')
    fullname_trainer = filename + '_trainer_state' + ext
    torch.save(checkpoint, fullname_trainer)


# This script is used to split pth files which generated before version 1.3.1 into two files.
# there is only one argument: --dir, fill the dir contains the pth files inside.
# NOTE: If you are using this script to convert the checkpoints of GPT3 or other sharding models,
# please rename the checkpoint filenames after the conversion manually.
parser = argparse.ArgumentParser()
parser.add_argument('--dir', help='The dir contains the *.pth files.')
args = parser.parse_args()
folder = args.dir
assert folder

all_files = os.listdir(folder)
all_files = [file for file in all_files if file.endswith('.pth')]
for file in all_files:
    shutil.copy(os.path.join(folder, file), os.path.join(folder, file + '.legacy'))
    convert_single_pth(os.path.join(folder, file))
