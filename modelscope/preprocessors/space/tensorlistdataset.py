#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.utils.data import Dataset


class TensorListDataset(Dataset):
    r"""Dataset wrapping tensors, tensor dicts and tensor lists.

    Arguments:
        *data (Tensor or dict or list of Tensors): tensors that have the same size
        of the first dimension.
    """

    def __init__(self, *data):
        if isinstance(data[0], dict):
            size = list(data[0].values())[0].size(0)
        elif isinstance(data[0], list):
            size = data[0][0].size(0)
        else:
            size = data[0].size(0)
        for element in data:
            if isinstance(element, dict):
                assert all(
                    size == tensor.size(0)
                    for name, tensor in element.items())  # dict of tensors
            elif isinstance(element, list):
                assert all(size == tensor.size(0)
                           for tensor in element)  # list of tensors
            else:
                assert size == element.size(0)  # tensor
        self.size = size
        self.data = data

    def __getitem__(self, index):
        result = []
        for element in self.data:
            if isinstance(element, dict):
                result.append({k: v[index] for k, v in element.items()})
            elif isinstance(element, list):
                result.append(v[index] for v in element)
            else:
                result.append(element[index])
        return tuple(result)

    def __len__(self):
        return self.size
