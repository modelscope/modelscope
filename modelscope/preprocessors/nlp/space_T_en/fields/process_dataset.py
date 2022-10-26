# Copyright (c) rhythmcao modified from https://github.com/rhythmcao/text2sql-lgesql.

import os
import pickle
import sys

from text2sql_lgesql.asdl.asdl import ASDLGrammar
from text2sql_lgesql.asdl.transition_system import TransitionSystem

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def process_example(processor, entry, db, trans, verbose=False):
    # preprocess raw tokens, schema linking and subgraph extraction
    entry = processor.pipeline(entry, db, verbose=verbose)
    # generate target output actions
    entry['ast'] = []
    entry['actions'] = []
    return entry


def process_tables(processor, tables_list, output_path=None, verbose=False):
    tables = {}
    for each in tables_list:
        if verbose:
            print('*************** Processing database %s **************' %
                  (each['db_id']))
        tables[each['db_id']] = processor.preprocess_database(
            each, verbose=verbose)
    print('In total, process %d databases .' % (len(tables)))
    if output_path is not None:
        pickle.dump(tables, open(output_path, 'wb'))
    return tables


def process_dataset(model_dir,
                    processor,
                    dataset,
                    tables,
                    output_path=None,
                    skip_large=False,
                    verbose=False):
    grammar = ASDLGrammar.from_filepath(
        os.path.join(model_dir, 'sql_asdl_v2.txt'))
    trans = TransitionSystem.get_class_by_lang('sql')(grammar)
    processed_dataset = []
    for idx, entry in enumerate(dataset):
        if skip_large and len(tables[entry['db_id']]['column_names']) > 100:
            continue
        if verbose:
            print('*************** Processing %d-th sample **************' %
                  (idx))
        entry = process_example(
            processor, entry, tables[entry['db_id']], trans, verbose=verbose)
        processed_dataset.append(entry)
    if output_path is not None:
        # serialize preprocessed dataset
        pickle.dump(processed_dataset, open(output_path, 'wb'))
    return processed_dataset
