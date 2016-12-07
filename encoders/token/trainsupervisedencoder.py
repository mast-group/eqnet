import os

from experimenter import ExperimentLogger

from data.utils import file_md5
from encoders.evaluation.knnstats import SemanticEquivalentDistanceEvaluation
from encoders.token.grusupervisedencoder import SequenceGruSupervisedEncoder

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 6 and len(sys.argv) != 5:
        print(
            "Usage <trainFile> <validationFile> <testFile> <isNoiseSemanticallyEquivalent> [OPTIONAL <outputFolderName>]")
        sys.exit(-1)

    # Supervised Centroid GRU Parameters
    # hyperparameters = dict(log_learning_rate=-2.505912542,
    #                        rmsprop_rho=0.800023259,
    #                        momentum=0.479903172,
    #                        grad_clip=1.918860396,
    #                        minibatch_size=411,
    #                        embedding_size=64,
    #                        representation_size=32,
    #                        log_init_noise=-1.884359806,
    #                        dropout_rate=0.4,
    #                        num_centroids=498,
    #                        centroid_use_rate=0.18905478
    #                        )

    # Supervised GRU Parameters
    hyperparameters = dict(log_learning_rate=-2.317727933,
                           rmsprop_rho=0.900376565,
                           momentum=0.660722378,
                           grad_clip=0.867701675,
                           minibatch_size=100,
                           embedding_size=128,
                           representation_size=32,
                           log_init_noise=-1,
                           dropout_rate=0.260395478)
    training_file = sys.argv[1]
    validation_file = sys.argv[2]
    test_file = sys.argv[3]
    all_params = dict(hyperparameters)
    all_params["training_set"] = training_file
    all_params["validation_set"] = validation_file
    assert sys.argv[4] == 'True' or sys.argv[4] == 'False'
    semantically_equivalent_noise = sys.argv[4] == 'True'

    ae = SequenceGruSupervisedEncoder(training_file, hyperparameters)
    evaluation = SemanticEquivalentDistanceEvaluation('', ae)


    def calculate_knn_score():
        eval_results = evaluation.evaluate(test_file)
        print("Full kNN: %s" % eval_results)


    with ExperimentLogger(name="GRUSupervisedEncoder", parameters=all_params,
                          directory=os.path.dirname(__file__)) as experiment_logger:
        val_cross_entropy, historic_values = ae.train(training_file, validation_file,
                                                      semantically_equivalent_noise=semantically_equivalent_noise,
                                                      additional_code_to_run=calculate_knn_score)
        if len(historic_values) > 0:
            trained_file = os.path.basename(training_file)
            assert trained_file.endswith('-trainset.json.gz')
            pickled_filename = 'grusupervisedencoder-' + trained_file[:-len('-trainset.json.gz')] + '.pkl'
            if len(sys.argv) == 6:
                ae.save(sys.argv[5] + pickled_filename)
            else:
                ae.save(pickled_filename)
            pickled_filename_md5 = file_md5(pickled_filename)
            experiment_logger.record_results(
                {"validation_cross_entropy": val_cross_entropy, "entropy_history": historic_values,
                 "model_info": dict(name=pickled_filename, md5=pickled_filename_md5)})
