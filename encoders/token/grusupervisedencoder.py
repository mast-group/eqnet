import time
from itertools import chain

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import trange

from encoders.token.tokendata import TokenAutoencoderDatasetExtractor

theano.config.floatX = "float32"

from encoders.baseencoder import AbstractEncoder
from data.dataimport import import_data
from data.featuredict import get_empirical_distribution
from deeplearning.layers import GRU, AveragingGRU
from deeplearning.optimization import nesterov_rmsprop_multiple, log_softmax, dropout
from deeplearning.utils import Bunch


class SequenceGruSupervisedEncoderModel:
    """
    A sequence GRU supervised encoder
    """

    def __init__(self, embedding_size: int, vocabulary_size: int, empirical_distribution, representation_size: int,
                 hyperparameters: dict, encoder_type: str, name: str = "GRUSequenceSupervisedEncoder",
                 use_centroid=False):
        self.__hyperparameters = hyperparameters
        self.__name = name
        log_init_noise = self.__hyperparameters["log_init_noise"]

        self.__memory_size = representation_size
        self.__embedding_size = embedding_size

        embeddings = np.random.randn(vocabulary_size, embedding_size) * 10 ** log_init_noise
        self.__embeddings = theano.shared(embeddings.astype(theano.config.floatX), name=name + ":embeddings")
        self.__name_bias = theano.shared(np.log(empirical_distribution).astype(theano.config.floatX),
                                         name=name + ":name_bias")

        encoder_init_state = np.random.randn(representation_size) * 10 ** log_init_noise
        self.__encoder_init_state = theano.shared(encoder_init_state.astype(theano.config.floatX),
                                                  name=name + ":encoder_init_state")

        self.__rng = RandomStreams()

        self.__input_sequence = T.ivector(name + ":input_sequence")
        self.__output_sequence = T.ivector(name + ":output_sequence")
        self.__inverted_output_sequence = self.__output_sequence[::-1]
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
        self.__standalone_representation = T.dvector(self.__name + ":representation_input")

    @property
    def rng(self):
        return self.__rng

    @property
    def parameters(self):
        return self.__params

    @property
    def input_sequence_variable(self):
        return self.__input_sequence

    @property
    def output_sequence_variable(self):
        return self.__output_sequence

    @property
    def representation_variable(self):
        return self.__standalone_representation

    def get_encoding(self):
        """
        Return the encoding of the sequence.
        """
        encoded_rep = self.__encoder.get_encoding(self.__input_sequence, self.__encoder_init_state)
        return encoded_rep


class SequenceGruSupervisedEncoder(AbstractEncoder):
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
        self.__encoder = SequenceGruSupervisedEncoderModel(self.__hyperparameters["embedding_size"],
                                                           len(self.dataset_extractor.feature_map),
                                                           empirical_distribution,
                                                           self.__hyperparameters["representation_size"],
                                                           self.__hyperparameters, encoder_type=encoder_type,
                                                           use_centroid=use_centroid)

        target_embeddings = np.random.randn(self.__hyperparameters["representation_size"],
                                            self.dataset_extractor.num_equivalence_classes) * 10 ** \
                                                                                              self.__hyperparameters[
                                                                                                  "log_init_noise"]

        self.__target_embeddings = theano.shared(target_embeddings.astype(theano.config.floatX),
                                                 name="target_embeddings")
        self.__target_embeddings_dropout = dropout(self.__hyperparameters['dropout_rate'], self.__encoder.rng,
                                                   self.__target_embeddings, True)

        self.__trained_parameters = None
        self.__compiled_methods = None

    REQUIRED_HYPERPARAMETERS = {'log_learning_rate', 'rmsprop_rho', 'momentum', 'grad_clip', 'minibatch_size',
                                'embedding_size', 'representation_size', 'log_init_noise', 'dropout_rate'}

    def __get_loss(self, target_class, use_dropout):
        encoding = self.__encoder.get_encoding()
        target_embeddings = self.__target_embeddings_dropout if use_dropout else self.__target_embeddings
        logprobs = log_softmax(T.dot(encoding / encoding.norm(2), target_embeddings).dimshuffle('x', 0))[0]
        return logprobs, logprobs[target_class]

    def __compile_train_functions(self):
        target_class = T.iscalar(name="target_class")
        _, ll = self.__get_loss(target_class, True)

        wrt_vars = list(self.__encoder.parameters.values()) + [self.__target_embeddings]
        grad = T.grad(ll, wrt_vars)

        grad_acc = [theano.shared(np.zeros(param.get_value().shape).astype(theano.config.floatX)) for param in wrt_vars] \
                   + [theano.shared(0, name="sample_count")]
        self.__compiled_methods.grad_accumulate = theano.function(
            inputs=[self.__encoder.input_sequence_variable, target_class],
            updates=[(v, v + g) for v, g in zip(grad_acc, grad)] + [
                (grad_acc[-1], grad_acc[-1] + 1)],
            outputs=ll)

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
        target_class = T.iscalar(name="target_class")
        logprobs, ll = self.__get_loss(target_class, False)
        self.__compiled_methods.ll_and_logprobs = theano.function(
            inputs=[self.__encoder.input_sequence_variable, target_class],
            outputs=[ll, logprobs])

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
              validation_check_limit: int = 1, semantically_equivalent_noise: bool = False,
              additional_code_to_run=None) -> tuple:
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

        trainable_parameters = list(self.__encoder.parameters.values()) + [self.__target_embeddings]

        print("Num classes: %s" % self.dataset_extractor.num_equivalence_classes)

        def compute_validation_score() -> float:
            return compute_score(validation_set)

        def compute_score(dataset) -> float:
            # Get all encodings
            sum_ll = 0.
            correct = 0
            for data in dataset:
                ll, logprobs = self.__compiled_methods.ll_and_logprobs(data[0], data[2])
                sum_ll += ll
                if np.argmax(logprobs) == data[2]:
                    correct += 1
            print("Accuracy: %s" % (correct / len(dataset) * 100))
            return sum_ll / len(dataset)

        num_minibatches = max(1, min(int(np.floor(float(len(training_set)) / minibatch_size)), 25))  # Clump minibatches
        try:
            print("[%s] Training Started..." % time.asctime())
            ratios = np.zeros(len(trainable_parameters))
            n_batches = 0
            current_max_size = 3.
            curriculum_step = .2
            for i in range(max_iter):
                sample_ordering = []
                for j, tree_data in enumerate(training_set):
                    if tree_data[1] <= current_max_size:
                        sample_ordering.append(j)
                current_max_size += curriculum_step
                np.random.shuffle(np.array(sample_ordering, dtype=np.int32))
                n_batches = 0
                sum_train_loss = 0
                num_elements = 0

                for j in trange(num_minibatches, desc="Minibatch"):
                    for k in trange(j * minibatch_size, min((j + 1) * minibatch_size, len(sample_ordering)),
                                    desc="Sample", leave=False):
                        current_idx = sample_ordering[k]
                        loss = self.__compiled_methods.grad_accumulate(training_set[current_idx][0],
                                                                       training_set[current_idx][2])
                        sum_train_loss += loss
                        num_elements += 1

                    n_batches += 1
                    ratios += self.__compiled_methods.grad_step()

                if i % validation_check_limit == validation_check_limit - 1:
                    current_ll = compute_validation_score()
                    if current_ll > best_score:
                        best_score = current_ll
                        self.__save_current_params_as_best()
                        print("At %s validation: current_ll=%s [best so far]" % (i, current_ll))
                        epochs_not_improved = 0
                    else:
                        print("At %s validation: current_ll=%s" % (i, current_ll))
                        epochs_not_improved += 1

                    for k in range(len(trainable_parameters)):
                        print("%s: %.0e" % (trainable_parameters[k].name, ratios[k] / n_batches))

                    print("Train ll: %s" % (sum_train_loss / num_elements))
                    ratios = np.zeros_like(ratios)
                    if additional_code_to_run is not None: additional_code_to_run()

                if epochs_not_improved >= patience:
                    print("Not improved for %s epochs. Stopping..." % patience)
                    break

            print("[%s] Training Finished..." % time.asctime())
        except (InterruptedError, KeyboardInterrupt, SystemExit):
            print("Interrupted. Exiting training gracefully...")

        return best_score, historic_values

    def __save_current_params_as_best(self):
        self.__trained_parameters = [p.get_value() for p in
                                     list(self.__encoder.parameters.values()) + [self.__target_embeddings]]

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
