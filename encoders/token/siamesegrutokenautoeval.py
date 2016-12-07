import sys

import numpy as np

from encoders.evaluation.knnstats import SemanticEquivalentDistanceEvaluation
from encoders.token.grussiameseencoder import SequenceGruSiameseEncoder


def run_from_config(params, *args):
    if len(args) < 3:
        print("No input file or test file given: %s:%s" % (args, len(args)))
        sys.exit(-1)
    train_file = args[0]
    validation_file = args[1]
    test_file = args[2]

    local_params = dict(params)
    local_params["embedding_size"] = 2 ** local_params["log_embedding_size"]
    del local_params["log_embedding_size"]
    local_params["representation_size"] = 32  # Keep this fixed

    encoder = SequenceGruSiameseEncoder(train_file, local_params)
    evaluation = SemanticEquivalentDistanceEvaluation('', encoder)
    _ = encoder.train(train_file, validation_file)
    eval_results = np.sum(evaluation.evaluate(test_file))
    return -eval_results


if __name__ == '__main__':
    # Fast run, not necessarily good parameters
    params = dict()  # TODO: Add parameters
    print(run_from_config(params, *sys.argv[1:]))
