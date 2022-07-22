# Copyright (c) Alibaba, Inc. and its affiliates.

import inspect


def if_func_recieve_dict_inputs(func, inputs):
    """to decide if a func could recieve dict inputs or not

    Args:
        func (class): the target function to be inspected
        inputs (dicts): the inputs that will send to the function

    Returns:
        bool: if func recieve dict, then recieve True

    Examples:
        input = {"input_dict":xxx, "attention_masked":xxx},
            function(self, inputs) then return True
            function(inputs) then return True
            function(self, input_dict, attention_masked) then return False
    """
    signature = inspect.signature(func)
    func_inputs = list(signature.parameters.keys() - set(['self']))
    mismatched_inputs = list(set(func_inputs) - set(inputs))
    if len(func_inputs) == len(mismatched_inputs):
        return True
    else:
        return False
