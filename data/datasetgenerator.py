import gzip
import json
import os
import random
from collections import OrderedDict

TRAIN_SET = 0.6
VALIDATION_SET = 0.1

USE_DICT_DATA_STRUCT = True


def import_shuffle_data(arg_values):
    """
    Given arg values of input path and output path, this method does the process of concatenating and shuffling and then storing the output into three files for test, train and validate.
    :param arg_values:
    :return:
    """

    # Retreive compressed json files from all listed entries and concat them to create a single dictionary
    if USE_DICT_DATA_STRUCT:
        result = dict()
    else:
        result = list()
    for i in range(1, len(arg_values) - 1):
        path = arg_values[i]
        if os.path.isdir(path):
            for data in os.listdir(path):
                if (data.lower().endswith('.json.gz')):
                    if USE_DICT_DATA_STRUCT:
                        result.update(open_gzipped_json_data(data))
                    else:
                        result.extend(open_gzipped_json_data(path))
        elif (path.lower().endswith('.json.gz')):
            if USE_DICT_DATA_STRUCT:
                result.update(open_gzipped_json_data(path))
            else:
                result.extend(open_gzipped_json_data(path))

    if USE_DICT_DATA_STRUCT:
        # Retrieve keys and shuffle them
        keys = list(result)
        random.shuffle(keys)

        # Calculate threshold ranges for each sets
        keys_length = len(keys)
        train_set_threshold = int(keys_length * TRAIN_SET)
        validation_set_threshold = int(keys_length * VALIDATION_SET) + train_set_threshold

        # Extract seperate results from total result for each purpose
        train_set_result = extract_results_in_range(0, train_set_threshold, keys, result)
        validation_set_result = extract_results_in_range(train_set_threshold, validation_set_threshold, keys, result)
        test_set_result = extract_results_in_range(validation_set_threshold, keys_length, keys, result)
    else:
        # Calculate threshold ranges for each sets
        random.shuffle(result)
        keys_length = len(result)
        train_set_threshold = int(keys_length * TRAIN_SET)
        validation_set_threshold = int(keys_length * VALIDATION_SET) + train_set_threshold

        # Extract seperate results from total result for each purpose
        train_set_result = result[0:train_set_threshold]
        validation_set_result = result[train_set_threshold: validation_set_threshold]
        test_set_result = result[validation_set_threshold: keys_length]

    # Save each result set as a compressed json file
    save_result_as_gzipped_json(os.path.join(arg_values[-1], 'trainset.json.gz'), train_set_result)
    save_result_as_gzipped_json(os.path.join(arg_values[-1], 'validateset.json.gz'), validation_set_result)
    save_result_as_gzipped_json(os.path.join(arg_values[-1], 'testset.json.gz'), test_set_result)


def save_result_as_gzipped_json(result_path, result):
    """
    Given the path and result object, this method save a compressed json.gz file of result in given path
    :param result_path:
    :param result:
    :return:
    """
    with gzip.GzipFile(result_path, 'wb') as outfile:
        outfile.write(bytes(json.dumps(result), 'UTF-8'))


def open_gzipped_json_data(path_to_data: str):
    """
    Given an json.gz file this method opens it and return the content.
    :param path_to_data:
    :return:
    """
    with gzip.open(path_to_data, "rb") as f:
        return json.loads(f.read().decode("utf-8"), object_pairs_hook=OrderedDict)


def extract_results_in_range(start_index, end_index, keys, total_results):
    """
    Given the indexes and keys this method loop between indexes
    :param start_index:
    :param end_index:
    :param keys:
    :param total_results:
    :return:
    """
    return {k: total_results[k] for k in keys[start_index:end_index]}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage 1 <inputFolder> <outputFolder>")
        print("Usage 2 <inputJsonGz1> <inputJsonGz2> ... <outputFolder>")
        sys.exit(-1)
    import_shuffle_data(sys.argv)
