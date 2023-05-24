# Copyright (c) Alibaba, Inc. and its affiliates.

import math

import pandas as pd


class MaxComputeUtil:
    """
    MaxCompute util class.

    Args:
        access_id: your access id of MaxCompute
        access_key: access key of MaxCompute
        project_name: your project name of MaxCompute
        endpoint: endpoint of MaxCompute

    Attributes:
        _odps: ODPS object

    """

    def __init__(self, access_id, access_key, project_name, endpoint):
        from odps import ODPS
        self._odps = ODPS(access_id, access_key, project_name, endpoint)

    def _get_table(self, table_name):
        """
        Get MaxCompute table object.
        """
        return self._odps.get_table(table_name)

    def _read_data(self, table_name: str, pt_condition: str) -> pd.DataFrame:
        """
        Read data from MaxCompute table.
        :param table_name: table name
        :param pt_condition: partition condition,
            Example: pt_condition = 'dt=20230331'
        :return: pandas dataframe with all data
        """
        t = self._get_table(table_name)

        with t.open_reader(partition=pt_condition, limit=False) as reader:
            pd_df = reader.to_pandas()

        return pd_df

    def fetch_data_to_csv(self, table_name: str, pt_condition: str,
                          output_path: str) -> None:
        """
        Fetch data from MaxCompute table to local file.
        :param table_name: table name
        :param pt_condition: partition condition,
            Example: pt_condition = 'dt=20230331'
        :param output_path: output path
        :return: None
        """
        pd_df = self._read_data(table_name, pt_condition)
        pd_df.to_csv(output_path, index=False)
        print(f'Fetch data to {output_path} successfully.')

    @staticmethod
    def _check_batch_args(reader, batch_size, limit):
        if not limit:
            limit = reader.count
        if batch_size <= 0:
            raise ValueError(
                f'batch_size must be positive, but got {batch_size}')
        if batch_size > limit:
            batch_size = limit
        return batch_size, limit

    @staticmethod
    def gen_reader_batch(reader, batch_size_in: int, limit_in: int,
                         drop_last_in: bool, partitions: list, columns: list):
        """
        Generate batch data from MaxCompute table.

        Args:
            reader: MaxCompute table reader
            batch_size_in: batch size
            limit_in: limit of data, None means fetch all data
            drop_last_in: whether drop last incomplete batch data
            partitions: table partitions
            columns: table columns

        Returns:
            batch data generator
        """

        batch_size_in, limit_in = MaxComputeUtil._check_batch_args(
            reader, batch_size_in, limit_in)

        batch_num = math.floor(limit_in / batch_size_in)
        for i in range(batch_num + 1):
            if i == batch_num and not drop_last_in and limit_in % batch_size_in > 0:
                batch_records = reader[i * batch_size_in:(
                    i * batch_size_in + (limit_in % batch_size_in))]
            else:
                batch_records = reader[i * batch_size_in:(i + 1)
                                       * batch_size_in]
            batch_data_list = []
            for record in batch_records:
                tmp_vals = [val for _, val in list(record)]
                tmp_vals = tmp_vals[:(len(tmp_vals) - len(partitions))]
                batch_data_list.append(tmp_vals)
            yield pd.DataFrame(batch_data_list, columns=columns)

    @staticmethod
    def gen_reader_item(reader, index: int, batch_size_in: int, limit_in: int,
                        drop_last_in: bool, partitions: list, columns: list):
        """
        Get single batch data from MaxCompute table by indexing.

        Args:
            reader: MaxCompute table reader
            index: index of batch data
            batch_size_in: batch size
            limit_in: limit of data, None means fetch all data
            drop_last_in: whether drop last incomplete batch data
            partitions: table partitions
            columns: table columns

        Returns:
            single batch data (dataframe)
        """
        batch_size_in, limit_in = MaxComputeUtil._check_batch_args(
            reader, batch_size_in, limit_in)

        if drop_last_in:
            batch_num = math.floor(limit_in / batch_size_in)
        else:
            batch_num = math.ceil(limit_in / batch_size_in)

        if index < 0:
            raise ValueError(f'index must be non-negative, but got {index}')
        if index >= batch_num:
            raise ValueError(
                f'index must be less than batch_num, but got index={index}, batch_num={batch_num}'
            )

        start = index * batch_size_in
        end = (index + 1) * batch_size_in
        if end > limit_in:
            end = limit_in
        batch_item = reader[start:end]

        batch_data_list = []
        for record in batch_item:
            tmp_vals = [val for _, val in list(record)]
            tmp_vals = tmp_vals[:(len(tmp_vals) - len(partitions))]
            batch_data_list.append(tmp_vals)

        return pd.DataFrame(batch_data_list, columns=columns)

    def get_table_reader_ins(self, table_name: str, pt_condition: str = None):

        table_ins = self._get_table(table_name)
        with table_ins.open_reader(partition=pt_condition) as reader:
            return table_ins, reader
