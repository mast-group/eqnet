import random
import sys
from collections import defaultdict

from data.datasetgenerator import open_gzipped_json_data, save_result_as_gzipped_json


# Store back to .json.gz
def save_split(data: dict, filename: str):
    converted = {}
    for eq_class, samples in data.items():
        if len(samples) > 0:
            converted[eq_class] = dict(Original=samples[0], Noise=samples[1:])
    save_result_as_gzipped_json(filename, converted)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage <inputFile>")
        sys.exit(-1)

    # Load JSON
    input_filename = sys.argv[1]
    assert input_filename.endswith('.json.gz')
    base_path = input_filename[:-len('.json.gz')]
    all_data = open_gzipped_json_data(input_filename)

    # Put equivalence classes into buckets
    equivalence_classes = defaultdict(list)

    for eq_class, elements in all_data.items():
        equivalence_classes[eq_class] = [elements['Original']] + elements['Noise']

    avg_num_elements_per_class = sum(len(e) for e in equivalence_classes.values()) / len(equivalence_classes)

    # Split equivalence classes into train, val, test
    eqclass_train_split = .8
    eqclass_test_split = 1. - eqclass_train_split

    random.seed(123)
    eq_classes = list(equivalence_classes.keys())
    random.shuffle(eq_classes)

    neweq_test_classes = {}
    for eq_class in eq_classes:
        if len(equivalence_classes[eq_class]) > 3 * avg_num_elements_per_class or len(
                equivalence_classes[eq_class]) == 1:
            continue  # Do not use classes that have a large number of elements, since they will be easy. Do not use those that have exactly one, because we cannot do anything
        if len(neweq_test_classes) >= eqclass_test_split * len(equivalence_classes):
            break  # We gathered enough
        neweq_test_classes[eq_class] = equivalence_classes[eq_class]
        del equivalence_classes[eq_class]

    save_split(neweq_test_classes, base_path + '-neweqtestset.json.gz')

    train_pct = .6
    val_pct = .15
    test_pct = 1. - train_pct - val_pct
    assert train_pct > val_pct and train_pct > test_pct

    trainset = defaultdict(list)
    validationset = defaultdict(list)
    testset = defaultdict(list)

    for eq_class, samples in equivalence_classes.items():
        num_samples = len(samples)
        random.shuffle(samples)
        num_train_samples = int(num_samples * train_pct)
        num_val_samples = int(num_samples * val_pct)
        num_test_samples = int(num_samples * test_pct)
        if num_train_samples == 0:
            num_train_samples = num_samples

        # Add any residuals
        num_train_samples += num_samples - num_val_samples - num_train_samples - num_test_samples
        assert num_train_samples + num_val_samples + num_test_samples == num_samples, (
        num_train_samples, num_val_samples, num_test_samples, num_samples)
        trainset[eq_class] = samples[:num_train_samples]
        validationset[eq_class] = samples[num_train_samples:num_train_samples + num_val_samples]
        testset[eq_class] = samples[num_train_samples + num_val_samples:]

    save_split(trainset, base_path + '-trainset.json.gz')
    save_split(validationset, base_path + '-validationset.json.gz')
    save_split(testset, base_path + '-testset.json.gz')
