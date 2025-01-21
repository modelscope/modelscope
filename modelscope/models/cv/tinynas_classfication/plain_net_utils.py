# Copyright (c) Alibaba, Inc. and its affiliates.
# The ZenNAS implementation is also open-sourced by the authors, and available at https://github.com/idstcv/ZenNAS.

from torch import nn

from . import (basic_blocks, super_blocks, super_res_idwexkx, super_res_k1kxk1,
               super_res_kxkx)
from .global_utils import create_netblock_list_from_str_inner


class PlainNet(nn.Module):

    def __init__(self,
                 argv=None,
                 opt=None,
                 num_classes=None,
                 plainnet_struct=None,
                 no_create=False,
                 **kwargs):
        super(PlainNet, self).__init__()
        self.argv = argv
        self.opt = opt
        self.num_classes = num_classes
        self.plainnet_struct = plainnet_struct

        self.module_opt = None

        if self.num_classes is None:
            self.num_classes = self.module_opt.num_classes

        if self.plainnet_struct is None and self.module_opt.plainnet_struct is not None:
            self.plainnet_struct = self.module_opt.plainnet_struct

        if self.plainnet_struct is None:
            if hasattr(opt, 'plainnet_struct_txt'
                       ) and opt.plainnet_struct_txt is not None:
                plainnet_struct_txt = opt.plainnet_struct_txt
            else:
                plainnet_struct_txt = self.module_opt.plainnet_struct_txt

            if plainnet_struct_txt is not None:
                with open(plainnet_struct_txt, 'r', encoding='utf-8') as fid:
                    the_line = fid.readlines()[0].strip()
                    self.plainnet_struct = the_line
                pass

        if self.plainnet_struct is None:
            return

        the_s = self.plainnet_struct

        block_list, remaining_s = create_netblock_list_from_str_inner(
            the_s,
            netblocks_dict=_all_netblocks_dict_,
            no_create=no_create,
            **kwargs)
        assert len(remaining_s) == 0

        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)

    def forward(self, x):
        output = x
        for the_block in self.block_list:
            output = the_block(output)
        return output

    def __str__(self):
        s = ''
        for the_block in self.block_list:
            s += str(the_block)
        return s

    def __repr__(self):
        return str(self)


_all_netblocks_dict_ = {}
_all_netblocks_dict_ = basic_blocks.register_netblocks_dict(
    _all_netblocks_dict_)
_all_netblocks_dict_ = super_blocks.register_netblocks_dict(
    _all_netblocks_dict_)
_all_netblocks_dict_ = super_res_kxkx.register_netblocks_dict(
    _all_netblocks_dict_)
_all_netblocks_dict_ = super_res_k1kxk1.register_netblocks_dict(
    _all_netblocks_dict_)
_all_netblocks_dict_ = super_res_idwexkx.register_netblocks_dict(
    _all_netblocks_dict_)
