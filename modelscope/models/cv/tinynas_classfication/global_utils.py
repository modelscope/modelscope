# Copyright (c) Alibaba, Inc. and its affiliates.
# The ZenNAS implementation is also open-sourced by the authors, and available at https://github.com/idstcv/ZenNAS.


def smart_round(x, base=None):
    if base is None:
        if x > 32 * 8:
            round_base = 32
        elif x > 16 * 8:
            round_base = 16
        else:
            round_base = 8
    else:
        round_base = base

    return max(round_base, round(x / float(round_base)) * round_base)


def get_right_parentheses_index(s):
    left_paren_count = 0
    for index, x in enumerate(s):

        if x == '(':
            left_paren_count += 1
        elif x == ')':
            left_paren_count -= 1
            if left_paren_count == 0:
                return index
        else:
            pass
    return None


def create_netblock_list_from_str_inner(s,
                                        no_create=False,
                                        netblocks_dict=None,
                                        **kwargs):
    block_list = []
    while len(s) > 0:
        is_found_block_class = False
        for the_block_class_name in netblocks_dict.keys():
            tmp_idx = s.find('(')
            if tmp_idx > 0 and s[0:tmp_idx] == the_block_class_name:
                is_found_block_class = True
                the_block_class = netblocks_dict[the_block_class_name]
                the_block, remaining_s = the_block_class.create_from_str(
                    s, no_create=no_create, **kwargs)
                if the_block is not None:
                    block_list.append(the_block)
                s = remaining_s
                if len(s) > 0 and s[0] == ';':
                    return block_list, s[1:]
                break
        assert is_found_block_class
    return block_list, ''


def create_netblock_list_from_str(s,
                                  no_create=False,
                                  netblocks_dict=None,
                                  **kwargs):
    the_list, remaining_s = create_netblock_list_from_str_inner(
        s, no_create=no_create, netblocks_dict=netblocks_dict, **kwargs)
    assert len(remaining_s) == 0
    return the_list
