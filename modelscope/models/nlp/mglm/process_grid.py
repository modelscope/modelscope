# Copyright (c) 2022 Zhipu.AI

import glob
import os
import statistics
import sys

import json

path_pattern = sys.argv[1]
target_type = sys.argv[2]
best_value, best_result, best_name = None, None, None
mean_result = {}
print(path_pattern)
for dir_path in glob.glob(path_pattern, recursive=True):
    entry = os.path.basename(dir_path)
    valid_result = None
    test_found = os.path.exists(os.path.join(dir_path, 'test_results.json'))
    valid_path = os.path.join(dir_path, 'results.json')
    if os.path.exists(valid_path):
        print(entry)
        with open(valid_path, encoding='utf-8') as file:
            valid_result = json.load(file)
    else:
        print(f'{entry} no validation results')
        continue
    if not test_found:
        print(f'{entry} not tested yet')
    if target_type == 'max':
        metric = sys.argv[3]
        metric_value = valid_result[metric]
        if best_value is None or metric_value > best_value:
            best_value = metric_value
            best_result = valid_result
            best_name = entry
    elif target_type == 'mean' or target_type == 'median':
        if mean_result:
            for metric, value in valid_result.items():
                if metric not in ['type', 'epoch']:
                    mean_result[metric].append(value)
        else:
            mean_result = {
                metric: [value]
                for metric, value in valid_result.items()
                if metric not in ['type', 'epoch']
            }

if target_type == 'max':
    print(f'Best result found at {best_name}: {best_result}')
elif target_type == 'mean':
    mean_result = {
        metric: sum(value) / len(value)
        for metric, value in mean_result.items()
    }
    print(f'Mean result {mean_result}')
elif target_type == 'median':
    mean_result = {
        metric: statistics.median(value)
        for metric, value in mean_result.items()
    }
    print(f'Mean result {mean_result}')
