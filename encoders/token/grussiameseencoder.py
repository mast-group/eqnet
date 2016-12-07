import time
from collections import defaultdict
from itertools import chain, permutations

import numpy as np
import theano
from scipy.spatial.distance import pdist
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import trange

from encoders.token.tokendata import TokenAutoencoderDatasetExtractor

theano.config.floatX = "float32"

from encoders.baseencoder import AbstractEncoder
from data.dataimport import import_data
from data.featuredict import get_empirical_distribution
from deeplearning.layers import GRU, AveragingGRU
from deeplearning.optimization import nesterov_rmsprop_multiple
from deeplearning.utils import Bunch


class SequenceGruSiameseEncoderModel:
    """
    A sequence GRU siamese encoder
    """

    def __init__(self, embedding_size: int, vocabulary_size: int, empirical_distribution, representation_size: int,
                 hyperparameters: dict, encoder_type: str, name: str = "GRUSequenceSiameseEncoder", use_centroid=False):
        self.__hyperparameters = hyperparameters
        self.__name = name
        log_init_noise = self.__hyperparameters["log_init_noise"]

        self.__memory_size = representation_size
        self.__embedding_size = embedding_size
        self.__vocabulary_size = vocabulary_size
        self.__empirical_distribution = empirical_distribution
        self.__encoder_type = encoder_type

        embeddings = np.random.randn(vocabulary_size, embedding_size) * 10 ** log_init_noise
        self.__embeddings = theano.shared(embeddings.astype(theano.config.floatX), name=name + ":embeddings")
        self.__name_bias = theano.shared(np.log(empirical_distribution).astype(theano.config.floatX),
                                         name=name + ":name_bias")

        encoder_init_state = np.random.randn(representation_size) * 10 ** log_init_noise
        self.__encoder_init_state = theano.shared(encoder_init_state.astype(theano.config.floatX),
                                                  name=name + ":encoder_init_state")

        self.__rng = RandomStreams()

        self.__input_sequence = T.ivector(name + ":input_sequence")

        if encoder_type == 'gru':
            self.__encoder = GRU(self.__embeddings, representation_size, embedding_size,
                                 self.__hyperparameters, self.__rng, name=name + ":GRUSequenceEncoder",
                                 use_centroid=use_centroid)
        elif encoder_type == 'averaging_gru':
            self.__encoder = AveragingGRU(self.__embeddings, representation_size, embedding_size,
                                          self.__hyperparameters, self.__rng,
                                          name=name + ":AveragingGRUSequenceEncoder", use_centroid=use_centroid)
        else:
            raise Exception("Unrecognized encoder type `%s`, possible options `gru` and `averaging_gru`")

        self.__params = {"embeddings": self.__embeddings,
                         "encoder_init_state": self.__encoder_init_state}
        self.__params.update(self.__encoder.get_params())

    @property
    def rng(self):
        return self.__rng

    @property
    def parameters(self):
        return self.__params

    @property
    def input_sequence_variable(self):
        return self.__input_sequence

    def get_encoding(self):
        """
        Return the encoding of the sequence.
        """
        encoded_rep = self.__encoder.get_encoding(self.__input_sequence, self.__encoder_init_state)
        return encoded_rep

    def copy_full(self, name):
        copy = SequenceGruSiameseEncoderModel(self.__embedding_size, self.__vocabulary_size,
                                              self.__empirical_distribution,
                                              self.__memory_size, self.__hyperparameters, self.__encoder_type,
                                              name=name)
        copy.__name_bias = self.__name_bias
        copy.__embeddings = self.__embeddings
        copy.__encoder_init_state = self.__encoder_init_state
        copy.__rng = self.__rng
        copy.__encoder = self.__encoder
        copy.__params = dict
        return copy


class SequenceGruSiameseEncoder(AbstractEncoder):
    """
    Train an encoder
    """

    def __init__(self, training_file, hyperparameters, encoder_type='gru', use_centroid=False):
        """

        :param training_file:
        :type hyperparameters: dict
        :return:
        """
        self.__hyperparameters = hyperparameters

        self.dataset_extractor = TokenAutoencoderDatasetExtractor(training_file)

        empirical_distribution = get_empirical_distribution(self.dataset_extractor.feature_map,
                                                            chain(*self.dataset_extractor.get_nonnoisy_samples(
                                                                import_data(training_file))))
        self.__encoder = SequenceGruSiameseEncoderModel(self.__hyperparameters["embedding_size"],
                                                        len(self.dataset_extractor.feature_map),
                                                        empirical_distribution,
                                                        self.__hyperparameters["representation_size"],
                                                        self.__hyperparameters, encoder_type=encoder_type,
                                                        use_centroid=use_centroid)

        self.__trained_parameters = None
        self.__compiled_methods = None

    REQUIRED_HYPERPARAMETERS = {'log_learning_rate', 'rmsprop_rho', 'momentum', 'grad_clip', 'minibatch_size',
                                'embedding_size', 'representation_size', 'log_init_noise', 'dropout_rate'}

    def __get_siamese_loss(self, use_dropout, scale_similar=1, scale_dissimilar=1):
        encoder_copy = self.__encoder.copy_full(name="siameseEncoder")
        encoding_1 = self.__encoder.get_encoding()
        encoding_2 = encoder_copy.get_encoding()

        representation_distance = (encoding_1 - encoding_2).norm(2)
        similar_loss = -scale_similar * T.pow(representation_distance, 2)
        margin = self.__hyperparameters['dissimilar_margin']
        dissimilar_loss = -scale_dissimilar * T.pow(T.nnet.relu(margin - representation_distance), 2)
        return dissimilar_loss, similar_loss, encoder_copy, encoding_1, encoding_2

    def __compile_train_functions(self):
        dissimilar_loss, similar_loss, encoder_copy, repr1, repr2 = self.__get_siamese_loss(True)

        wrt_vars = list(self.__encoder.parameters.values())

        grad_acc = [theano.shared(np.zeros(param.get_value().shape).astype(theano.config.floatX)) for param in wrt_vars] \
                   + [theano.shared(0, name="sample_count")]

        grad = T.grad(similar_loss, wrt_vars)
        self.__compiled_methods.grad_siamese_similar = theano.function(
            inputs=[encoder_copy.input_sequence_variable, self.__encoder.input_sequence_variable],
            updates=[(v, v + g) for v, g in zip(grad_acc, grad)] + [
                (grad_acc[-1], grad_acc[-1] + 1)],
            outputs=[similar_loss, repr1, repr2])

        grad = T.grad(dissimilar_loss, wrt_vars)
        self.__compiled_methods.grad_siamese_dissimilar = theano.function(
            inputs=[encoder_copy.input_sequence_variable, self.__encoder.input_sequence_variable],
            updates=[(v, v + g) for v, g in zip(grad_acc, grad)] + [
                (grad_acc[-1], grad_acc[-1] + 1)],
            outputs=[dissimilar_loss, repr1, repr2])

        normalized_grads = [T.switch(grad_acc[-1] > 0, g / grad_acc[-1].astype(theano.config.floatX), g) for g in
                            grad_acc[:-1]]
        step_updates, ratios = nesterov_rmsprop_multiple(wrt_vars, normalized_grads,
                                                         learning_rate=10 ** self.__hyperparameters[
                                                             "log_learning_rate"],
                                                         rho=self.__hyperparameters["rmsprop_rho"],
                                                         momentum=self.__hyperparameters["momentum"],
                                                         grad_clip=self.__hyperparameters["grad_clip"],
                                                         output_ratios=True)
        step_updates.extend([(v, T.zeros(v.shape)) for v in grad_acc[:-1]])  # Set accumulators to 0
        step_updates.append((grad_acc[-1], 0))

        self.__compiled_methods.grad_step = theano.function(inputs=[], updates=step_updates, outputs=ratios)

    def __compile_test_functions(self):
        dissimilar_loss, similar_loss, encoder_copy, _, _ = self.__get_siamese_loss(False)
        self.__compiled_methods.test_similar_loss = theano.function(
            inputs=[encoder_copy.input_sequence_variable, self.__encoder.input_sequence_variable], outputs=similar_loss)
        self.__compiled_methods.test_dissimilar_loss = theano.function(
            inputs=[encoder_copy.input_sequence_variable, self.__encoder.input_sequence_variable],
            outputs=dissimilar_loss)

        self.__compiled_methods.encode = theano.function(inputs=[self.__encoder.input_sequence_variable],
                                                         outputs=self.__encoder.get_encoding())

    def __compile_if_needed(self):
        if self.__compiled_methods is None:
            print("Compiling Methods...")
            self.__compiled_methods = Bunch()
            self.__compile_train_functions()
            self.__compile_test_functions()
            print("Compilation Finished...")

    def train(self, training_file: str, validation_file: str, max_iter: int = 1000, patience: int = 25,
              validation_check_limit: int = 1, additional_code_to_run=None) -> tuple:
        self.__compile_if_needed()

        minibatch_size = self.__hyperparameters["minibatch_size"]
        training_data = import_data(training_file)
        training_set = list(self.dataset_extractor.get_dataset_for_encoder(training_data, return_num_tokens=True))
        validation_set = list(
            self.dataset_extractor.get_dataset_for_encoder(import_data(validation_file), return_num_tokens=True))
        best_score = float('-inf')
        train_x_ent = 0
        epochs_not_improved = 0
        historic_values = []

        trainable_parameters = list(self.__encoder.parameters.values())

        print("Num classes: %s" % self.dataset_extractor.num_equivalence_classes)

        def compute_validation_score() -> float:
            return compute_score(validation_set)

        def compute_score(dataset) -> float:
            # Get all encodings
            encodings = []
            equivalents = defaultdict(set)
            for i, tree in enumerate(dataset):
                encodings.append(self.__compiled_methods.encode(tree[0]))
                equivalents[tree[2]].add(i)

            encodings = np.array(encodings, dtype=theano.config.floatX)
            distances = pdist(encodings, metric='euclidean')

            is_similar = np.zeros_like(distances, dtype=np.int)
            for equivalence_set in equivalents.values():
                for i, j in permutations(equivalence_set, 2):
                    if i > j:
                        is_similar[encodings.shape[0] * j - int(j * (j + 1) / 2) + i - 1 - j] = 1

            similar_score = -np.sum(np.power(distances * is_similar, 2))

            margin = self.__hyperparameters['dissimilar_margin']
            differences = margin - distances
            rectified_diffs = differences * (differences > 0)
            dissimilar_score = -np.sum(np.power(rectified_diffs * (1 - is_similar), 2))

            print("Similar Loss: %s  Dissimilar Loss: %s" % (-similar_score, -dissimilar_score))
            return similar_score + dissimilar_score

        if self.__trained_parameters is None:
            best_score = float('-inf')
        else:
            best_score = compute_validation_score()
            print("Previous best validation score: %s" % best_score)

        try:
            print("[%s] Training Started..." % time.asctime())
            sum_similar_loss = 0
            num_similar_loss = 0
            sum_dissimilar_loss = 0
            num_dissimilar_loss = 0
            ratios = np.zeros(len(list(self.__encoder.parameters.values())))
            epochs_not_improved = 0
            # Clump minibatches and disallow minibatches that are smaller than their given size, since they may
            # cause instability.
            num_minibatches = max(1, min(int(np.floor(float(len(training_set)) / minibatch_size)), 2))

            current_max_size = 4.
            curriculum_step = .1

            for i in range(max_iter):
                sample_ordering = []
                for j, tree in enumerate(training_set):
                    if tree[-1] <= current_max_size:
                        sample_ordering.append(j)
                current_max_size += curriculum_step
                np.random.shuffle(np.array(sample_ordering, dtype=np.int32))
                n_batches = 0

                for j in trange(num_minibatches, desc="Minibatch"):
                    for k in trange(j * minibatch_size, min((j + 1) * minibatch_size, len(sample_ordering)),
                                    desc="Sample", leave=False):
                        current_idx = sample_ordering[k]
                        # Add siamese gradients, by picking num_examples
                        num_examples = 1  # The max number of examples to pick from TODO: as parameter
                        similar_snippet_idxs = []
                        dissimilar_snippet_idxs = []
                        for l in range(len(sample_ordering)):
                            if l == k:
                                continue
                            other_idx = sample_ordering[l]
                            if training_set[current_idx][2] == training_set[other_idx][2]:
                                similar_snippet_idxs.append(other_idx)
                            else:
                                dissimilar_snippet_idxs.append(other_idx)
                        dissimilar_snippet_idxs = np.array(dissimilar_snippet_idxs)

                        np.random.shuffle(similar_snippet_idxs)
                        for other_idx in similar_snippet_idxs:
                            loss, repr1, repr2 = self.__compiled_methods.grad_siamese_similar(
                                list(training_set[current_idx][0]), list(training_set[other_idx][0]))
                            sum_similar_loss += loss
                            num_similar_loss += 1

                        for other_idx in dissimilar_snippet_idxs:
                            loss, repr1, repr2 = self.__compiled_methods.grad_siamese_dissimilar(
                                training_set[current_idx][0], training_set[other_idx][0])
                            sum_dissimilar_loss += loss
                            num_dissimilar_loss += 1 if loss < 0 else 0

                    n_batches += 1
                    ratios += self.__compiled_methods.grad_step()

                if i % validation_check_limit == validation_check_limit - 1:
                    print("Iteration %s Stats" % i)
                    current_score = compute_validation_score()
                    if current_score > best_score:
                        best_score = current_score
                        self.__trained_parameters = [p.get_value() for p in list(self.__encoder.parameters.values())]
                        print("At %s validation: current_score=%s [best so far]" % (i, current_score))
                        epochs_not_improved = 0
                    else:
                        print("At %s validation: current_score=%s" % (i, current_score))
                        epochs_not_improved += 1
                    for k in range(len(list(self.__encoder.parameters.values()))):
                        print("%s: %.0e" % (list(self.__encoder.parameters.values())[k].name, ratios[k] / n_batches))

                    print("Train sum similar-loss: %s (%s samples)" % (sum_similar_loss, num_similar_loss))
                    print("Train sum dissimilar-loss: %s (%s samples)" % (sum_dissimilar_loss, num_dissimilar_loss))
                    print("Training Set stats: %s" % compute_score(training_set[:500]))

                    historic_values.append({"validation_xent": current_score})

                    sum_similar_loss = 0
                    num_similar_loss = 0
                    sum_dissimilar_loss = 0
                    num_dissimilar_loss = 0
                    ratios = np.zeros_like(ratios)
                    if additional_code_to_run is not None: additional_code_to_run()
                if epochs_not_improved >= patience:
                    print("Not improved for %s epochs. Stopping..." % patience)
                    break
            print("[%s] Training Finished..." % time.asctime())
        except (InterruptedError, KeyboardInterrupt):
            print("Interrupted. Exiting training gracefully...")

        return best_score, historic_values

    def __save_current_params_as_best(self):
        self.__trained_parameters = [p.get_value() for p in list(self.__encoder.parameters.values())]

    def save(self, filename: str):
        tmp, self.__compiled_methods = self.__compiled_methods, None
        AbstractEncoder.save(self, filename)
        self.__compiled_methods = tmp

    def get_representation_vector_size(self) -> int:
        return self.__hyperparameters["representation_size"]

    def get_encoding(self, data: tuple) -> np.array:
        self.__compile_if_needed()
        converted_tokens = self.dataset_extractor.tokens_to_array(data[0])
        return self.__compiled_methods.encode(converted_tokens)

    def decoder_loss(self, data: tuple, representation: np.array) -> float:
        raise NotImplementedError("An encoder cannot do this operation")
