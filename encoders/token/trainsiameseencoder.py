import os

from experimenter import ExperimentLogger

from data.utils import file_md5
from encoders.evaluation.knnstats import SemanticEquivalentDistanceEvaluation
from encoders.token.grussiameseencoder import SequenceGruSiameseEncoder

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4 and len(sys.argv) != 3:
        print("Usage <trainFile> <validationFile> [OPTIONAL <outputFolderName>]")
        sys.exit(-1)
    # Siamese GRU parameters
    # hyperparameters = dict(log_learning_rate=-3.5,
    #                        rmsprop_rho=0.8,
    #                        momentum=0,
    #                        grad_clip=0.2,
    #                        minibatch_size=999,
    #                        embedding_size=16,
    #                        representation_size=64,
    #                        log_init_noise=-5,
    #                        dropout_rate=0.4,
    #                        dissimilar_margin=0.1)

    # Siamese Centroid GRU parameters
    hyperparameters = dict(log_learning_rate=-0.806702998,
                           rmsprop_rho=0.953710675,
                           momentum=0,
                           grad_clip=0.677113327,
                           minibatch_size=999,
                           embedding_size=32,
                           representation_size=64,
                           log_init_noise=-5,
                           dropout_rate=0.121825398,
                           dissimilar_margin=0.1,
                           num_centroids=11,
                           centroid_use_rate=0.555898759)
    training_file = sys.argv[1]
    validation_file = sys.argv[2]
    all_params = dict(hyperparameters)
    all_params["training_set"] = training_file
    all_params["validation_set"] = validation_file

    ae = SequenceGruSiameseEncoder(training_file, hyperparameters, use_centroid=True)
    evaluation = SemanticEquivalentDistanceEvaluation('', ae)


    def calculate_knn_score():
        eval_results = evaluation.evaluate(sys.argv[2])
        print("Full kNN: %s" % eval_results)


    with ExperimentLogger(name="GRUSiameseEncoder", parameters=all_params,
                          directory=os.path.dirname(__file__)) as experiment_logger:
        val_cross_entropy, historic_values = ae.train(training_file, validation_file,
                                                      additional_code_to_run=None)
        if len(historic_values) > 0:
            trained_file = os.path.basename(training_file)
            assert trained_file.endswith('-trainset.json.gz')
            pickled_filename = 'grusiameseencoder-' + trained_file[:-len('-trainset.json.gz')] + '.pkl'
            if len(sys.argv) == 4:
                ae.save(sys.argv[3] + pickled_filename)
            else:
                ae.save(pickled_filename)
            pickled_filename_md5 = file_md5(pickled_filename)
            experiment_logger.record_results(
                {"validation_cross_entropy": val_cross_entropy, "entropy_history": historic_values,
                 "model_info": dict(name=pickled_filename, md5=pickled_filename_md5)})
