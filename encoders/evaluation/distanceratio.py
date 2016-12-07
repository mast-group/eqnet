import numpy as np
from scipy.spatial.distance import squareform, pdist

from data.dataimport import import_data
from encoders.baseencoder import AbstractEncoder


def get_representation_distance_ratio(encoder: AbstractEncoder, data_filename: str, print_stats: bool = False):
    """Compute the ratio of the avg distance of points within an equivalence class vs the avg distance between all points"""
    data = import_data(data_filename)
    encodings = []
    equivalence_sets = []

    for name, code in data.items():
        idx = len(encodings)
        enc = encoder.get_encoding(code['original'])
        assert not np.isnan(np.sum(enc))
        encodings.append(enc)
        for noisy_sample in code['noise']:
            enc = encoder.get_encoding(noisy_sample)
            assert not np.isnan(np.sum(enc))
            encodings.append(enc)
        equivalence_sets.append(set(range(idx, len(encodings))))

    encodings = np.array(encodings)

    all_distances = squareform(pdist(encodings, 'cosine'))  # TODO: avoid square form somehow
    assert not np.any(np.isnan(all_distances))

    # Average the lower triangle of all_distances
    avg_distance_between_all_points = np.sum(np.tril(all_distances, k=-1)) / (len(encodings) * (len(encodings) - 1) / 2)

    sum_distance_within_eq_class = 0.
    num_pairs = 0
    for equiv_class_idxs in equivalence_sets:
        num_elements_in_class = len(equiv_class_idxs)
        if num_elements_in_class < 2:
            continue
        elems_in_eq_class = np.fromiter(equiv_class_idxs, dtype=np.int32)
        sum_distance_within_eq_class += np.sum(np.tril(all_distances[elems_in_eq_class][:, elems_in_eq_class], k=-1))
        num_pairs += num_elements_in_class * (num_elements_in_class - 1) / 2

    avg_distance_within_eq_class = sum_distance_within_eq_class / num_pairs
    if print_stats:
        print(
            "Within Avg Dist: %s  All Avg Dist: %s " % (avg_distance_within_eq_class, avg_distance_between_all_points))
    return avg_distance_between_all_points / avg_distance_within_eq_class


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage <encoderFilename> <evalFilename>")
        sys.exit(-1)
    encoder = AbstractEncoder.load(sys.argv[1])
    ratio = get_representation_distance_ratio(encoder, sys.argv[2], True)
    print("Distance Ratio: %s" % ratio)
