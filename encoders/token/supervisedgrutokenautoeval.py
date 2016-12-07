import sys

import numpy as np

from encoders.evaluation.knnstats import SemanticEquivalentDistanceEvaluation
from encoders.token.grusupervisedencoder import SequenceGruSupervisedEncoder


def run_from_config(params, *args):
    if len(args) < 3:
        print("Usage: <train_file> <validation_file> <test_file> [<all_file>]  %s:%s" % (args, len(args)))
        sys.exit(-1)
    train_file = args[0]
    validation_file = args[1]
    test_file = args[2]

    local_params = dict(params)
    local_params["embedding_size"] = 2 ** local_params["log_embedding_size"]
    del local_params["log_embedding_size"]
    local_params["representation_size"] = 32  # Keep this fixed

    encoder = SequenceGruSupervisedEncoder(train_file, local_params, use_centroid=True)
    evaluation = SemanticEquivalentDistanceEvaluation('', encoder)
    val_xentropy, _ = encoder.train(train_file, validation_file)
    if len(args) > 3:
        eval_results = np.sum(evaluation.evaluate_with_test(test_file, args[3]))
    else:
        eval_results = np.sum(evaluation.evaluate(test_file))
    return -eval_results


if __name__ == '__main__':
    # Fast run, not necessarily good parameters
    params = dict()  # TODO: Add parameters
    print(run_from_config(params, *sys.argv[1:]))
