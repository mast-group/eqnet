import pickle

from experimenter import ExperimentLogger

from encoders.evaluation.knnstats import SemanticEquivalentDistanceEvaluation
from encoders.rnn.siameseencoder import RecursiveNNSiameseEncoder

if __name__ == '__main__':
    import sys
    import os

    if len(sys.argv) != 3:
        print("Usage <trainingFileName> <validationFileName>")
        sys.exit(-1)
    hyperparameters = dict(log_learning_rate=-2.41,
                           rmsprop_rho=.93,
                           momentum=0.59,
                           minibatch_size=1000,
                           memory_size=32,
                           ae_representation_size=16,
                           ae_noise=.5,
                           grad_clip=2,
                           log_init_scale_embedding=-3,
                           dropout_rate=0,
                           hidden_layer_sizes=[32],
                           constrain_intro_rate=.99,
                           curriculum_initial_size=4,
                           curriculum_step=.2,
                           dissimilar_margin=.1,
                           max_num_similar_examples=2,
                           max_num_dissimilar_examples=3)

    training_set = sys.argv[1]
    trained_file = os.path.basename(training_set)
    # assert trained_file.endswith('-trainset.json.gz')
    validation_set = sys.argv[2]
    all_params = dict(hyperparameters)
    all_params["training_set"] = training_set
    all_params["validation_set"] = validation_set
    encoder = RecursiveNNSiameseEncoder(training_set, hyperparameters)
    evaluation = SemanticEquivalentDistanceEvaluation('', encoder)


    def store_knn_score(historic_data: dict):
        eval_results = evaluation.evaluate(sys.argv[2])
        print("Full kNN: %s" % eval_results)
        historic_data['kNNeval'].append(eval_results)


    with ExperimentLogger(name="TreeRnnEncoder", parameters=all_params,
                          directory=os.path.dirname(__file__)) as experiment_logger:
        validation_score, historic_data = encoder.train(training_set, validation_set,
                                                        additional_code_to_run=store_knn_score)
        pickled_filename = 'rnnencoder-' + trained_file[:-len('-trainset.json.gz')] + '.pkl'
        encoder.save(pickled_filename)
        with open('historic-data' + pickled_filename, 'wb') as f:
            pickle.dump(historic_data, f)
        experiment_logger.record_results(
            {"validation_cross_entropy": validation_score, "model_info": dict(name=pickled_filename)})
