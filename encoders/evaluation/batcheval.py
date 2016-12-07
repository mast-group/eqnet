import os

from encoders.baseencoder import AbstractEncoder
from encoders.evaluation.distanceratio import get_representation_distance_ratio
from encoders.evaluation.knnstats import SemanticEquivalentDistanceEvaluation


def evaluate_on_all_dims(encoder_filename: str, full_dataset_filename, test_datsets_fileprefix) -> dict:
    """Return a dict with all results from comparison"""
    encoder = AbstractEncoder.load(encoder_filename)

    testset_filename = test_datsets_fileprefix + '-testset.json.gz'
    assert os.path.exists(testset_filename)

    neweq_testset_filename = test_datsets_fileprefix + '-neweqtestset.json.gz'
    assert os.path.exists(neweq_testset_filename)

    results = {}
    results['testintradist'] = get_representation_distance_ratio(encoder, testset_filename)
    results['neweqintradist'] = get_representation_distance_ratio(encoder,
                                                                  neweq_testset_filename)

    nn_evaluator = SemanticEquivalentDistanceEvaluation(None, encoder)

    test_nn_all_stats = nn_evaluator.evaluate_with_test(full_dataset_filename, testset_filename, num_nns=15)
    test_nn_within_stats = nn_evaluator.evaluate_with_test(testset_filename,
                                                           testset_filename, num_nns=15)

    neweq_nn_all_stats = nn_evaluator.evaluate_with_test(full_dataset_filename, neweq_testset_filename, num_nns=15)
    neweq_nn_within_stats = nn_evaluator.evaluate_with_test(neweq_testset_filename,
                                                            neweq_testset_filename, num_nns=15)

    for i in range(15):
        results['testsetknn' + str(i + 1) + 'all'] = test_nn_all_stats[i]
        results['testsetknn' + str(i + 1) + 'within'] = test_nn_within_stats[i]
        results['neweqknn' + str(i + 1) + 'all'] = neweq_nn_all_stats[i]
        results['neweqknn' + str(i + 1) + 'within'] = neweq_nn_within_stats[i]

    return results
