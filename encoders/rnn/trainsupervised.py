import pickle

from experimenter import ExperimentLogger

from encoders.evaluation.knnstats import SemanticEquivalentDistanceEvaluation
from encoders.rnn.supervisedencoder import RecursiveNNSupervisedEncoder

if __name__ == '__main__':
    import sys
    import os

    if len(sys.argv) != 3:
        print("Usage <trainingFileName> <validationFileName>")
        sys.exit(-1)

    # Single Supervised Parameters
    # hyperparameters = dict(log_learning_rate=-3.5,
    #                        rmsprop_rho=0.694291521,
    #                        momentum=0.842054804,
    #                        minibatch_size=673,
    #                        memory_size=16,
    #                        grad_clip=0.483937549,
    #                        log_init_scale_embedding=-1,
    #                        dropout_rate=0,
    #                        curriculum_initial_size=6.782536363,
    #                        curriculum_step=0.435723594)

    hyperparameters = dict(log_learning_rate=-2.1,
                           rmsprop_rho=.88,
                           momentum=0.88,
                           minibatch_size=900,
                           memory_size=64,
                           ae_representation_size=8,
                           ae_noise=.61,
                           grad_clip=1.82,
                           log_init_scale_embedding=-2.05,
                           dropout_rate=0.11,
                           hidden_layer_sizes=[8],
                           constrain_intro_rate=.9999,
                           curriculum_initial_size=6.96,
                           curriculum_step=2.72,
                           accuracy_margin=.5)

    training_set = sys.argv[1]
    trained_file = os.path.basename(training_set)
    # assert trained_file.endswith('-trainset.json.gz')
    validation_set = sys.argv[2]
    all_params = dict(hyperparameters)
    all_params["training_set"] = training_set
    all_params["validation_set"] = validation_set
    encoder = RecursiveNNSupervisedEncoder(training_set, hyperparameters)
    evaluation = SemanticEquivalentDistanceEvaluation('', encoder)


    def store_knn_score(historic_data: dict):
        eval_results = evaluation.evaluate(sys.argv[2])
        print("Full kNN: %s" % eval_results)
        historic_data['kNNeval'].append(eval_results)


    with ExperimentLogger(name="TreeRnnSupervisedEncoder", parameters=all_params,
                          directory=os.path.dirname(__file__)) as experiment_logger:
        validation_score, historic_data = encoder.train(training_set, validation_set,
                                                        additional_code_to_run=store_knn_score)
        pickled_filename = 'rnnsupervisedencoder-' + trained_file[:-len('-trainset.json.gz')] + '.pkl'
        encoder.save(pickled_filename)
        with open('historic-data' + pickled_filename, 'wb') as f:
            pickle.dump(historic_data, f)
        experiment_logger.record_results(
            {"validation_cross_entropy": validation_score, "model_info": dict(name=pickled_filename)})
