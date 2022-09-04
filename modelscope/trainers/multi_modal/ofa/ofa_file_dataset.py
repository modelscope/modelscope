# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import os
import pickle

import torch


class OFAFileDataset:

    def __init__(self,
                 file_path,
                 selected_col_ids=None,
                 dtypes=None,
                 separator='\t',
                 cached_index=False):
        self.file_path = file_path
        assert os.path.exists(
            self.file_path), 'Error: The local datafile {} not exists!'.format(
                self.file_path)

        self.separator = separator
        if selected_col_ids is None:
            # default to all fields
            self.selected_col_ids = list(
                range(
                    len(
                        open(self.file_path).readline().rstrip('\n').split(
                            self.separator))))
        else:
            self.selected_col_ids = [
                int(col_id) for col_id in selected_col_ids.split(',')
            ]
        if dtypes is None:
            # default to str
            self.dtypes = [str for col_id in self.selected_col_ids]
        else:
            self.dtypes = [eval(col_dtype) for col_dtype in dtypes.split(',')]
            assert len(self.dtypes) == len(self.selected_col_ids)

        self.data_cnt = 0
        try:
            self.slice_id = torch.distributed.get_rank()
            self.slice_count = torch.distributed.get_world_size()
        except Exception:
            self.slice_id = 0
            self.slice_count = 1
        self.cached_index = cached_index
        self._init_seek_index()
        self._reader = self._get_reader()
        print('file {} slice_id {} row count {} total row count {}'.format(
            self.file_path, self.slice_id, self.row_count,
            self.total_row_count))

    def _init_seek_index(self):
        if self.cached_index:
            cache_path = '{}.index'.format(self.file_path)
            assert os.path.exists(
                cache_path), 'cache file {} not exists!'.format(cache_path)
            self.total_row_count, self.lineid_to_offset = pickle.load(
                open(cache_path, 'rb'))
            print(
                'local datafile {} slice_id {} use cached row_count and line_idx-to-offset mapping'
                .format(self.file_path, self.slice_id))
        else:
            # make an iteration over the file to get row_count and line_idx-to-offset mapping
            fp = open(self.file_path, 'r')
            print(
                'local datafile {} slice_id {} begin to initialize row_count and line_idx-to-offset mapping'
                .format(self.file_path, self.slice_id))
            self.total_row_count = 0
            offset = 0
            self.lineid_to_offset = []
            for line in fp:
                self.lineid_to_offset.append(offset)
                self.total_row_count += 1
                offset += len(line.encode('utf-8'))
            pickle.dump(self.lineid_to_offset,
                        open('{}.index'.format(self.file_path), 'wb'))
        self._compute_start_pos_and_row_count()
        print(
            'local datafile {} slice_id {} finished initializing row_count and line_idx-to-offset mapping'
            .format(self.file_path, self.slice_id))

    def _compute_start_pos_and_row_count(self):
        self.row_count = self.total_row_count // self.slice_count
        if self.slice_id < self.total_row_count - self.row_count * self.slice_count:
            self.row_count += 1
            self.start_pos = self.row_count * self.slice_id
        else:
            self.start_pos = self.row_count * self.slice_id + (
                self.total_row_count - self.row_count * self.slice_count)

    def _get_reader(self):
        fp = open(self.file_path, 'r')
        fp.seek(self.lineid_to_offset[self.start_pos])
        return fp

    def _seek(self, offset=0):
        try:
            print('slice_id {} seek offset {}'.format(self.slice_id,
                                                      self.start_pos + offset))
            self._reader.seek(self.lineid_to_offset[self.start_pos + offset])
            self.data_cnt = offset
        except Exception:
            print('slice_id {} seek offset {}'.format(self.slice_id, offset))
            self._reader.seek(self.lineid_to_offset[offset])
            self.data_cnt = offset

    def __del__(self):
        self._reader.close()

    def __len__(self):
        return self.row_count

    def get_total_row_count(self):
        return self.total_row_count

    def __getitem__(self, index):
        if self.data_cnt == self.row_count:
            print('reach the end of datafile, start a new reader')
            self.data_cnt = 0
            self._reader = self._get_reader()
        column_l = self._reader.readline().rstrip('\n').split(self.separator)
        self.data_cnt += 1
        column_l = [
            dtype(column_l[col_id])
            for col_id, dtype in zip(self.selected_col_ids, self.dtypes)
        ]
        return column_l
