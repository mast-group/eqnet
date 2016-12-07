import sys

import numpy as np

from encoders.evaluation.knnstats import SemanticEquivalentDistanceEvaluation
from encoders.rnn.supervisedencoder import RecursiveNNSupervisedEncoder


def run_from_config(params, *args):
    if len(args) < 5:
        print("No input file or test file given: %s:%s" % (args, len(args)))
        sys.exit(-1)
    train_file = args[0]
    validation_file = args[1]
    test_file = args[2]
    neweq_test_file = args[3]
    all_file = args[4]

    local_params = dict(params)
    local_params["memory_size"] = 2 ** local_params["log_memory_size"]
    del local_params["log_memory_size"]
    local_params["ae_representation_size"] = 2 ** local_params["log_ae_representation_size"]
    del local_params["log_ae_representation_size"]
    local_params["hidden_layer_sizes"] = [2 ** local_params["log_hidden_layer_size"]]
    del local_params["log_hidden_layer_size"]
    local_params["constrain_intro_rate"] = 1 - 10 ** -local_params["constrain_intro_log_rate"]
    del local_params["constrain_intro_log_rate"]

    encoder = RecursiveNNSupervisedEncoder(train_file, local_params, combination_type='residual_with_ae')
    evaluation = SemanticEquivalentDistanceEvaluation('', encoder)
    val_xentropy, _ = encoder.train(train_file, validation_file)
    eval_results = np.sum(evaluation.evaluate_with_test(all_file, test_file))
    eval_results += np.sum(evaluation.evaluate_with_test(all_file, neweq_test_file))
    return -eval_results


if __name__ == '__main__':
    # Fast run, not necessarily good parameters
    params = dict()  # TODO: Add parameters
    print(run_from_config(params, *sys.argv[1:]))
