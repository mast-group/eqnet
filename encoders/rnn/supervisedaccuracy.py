import sys

from encoders.baseencoder import AbstractEncoder
from encoders.rnn.supervisedencoder import RecursiveNNSupervisedEncoder

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage <pklFile> <testFile>")
        sys.exit(-1)

    encoder = AbstractEncoder.load(sys.argv[1])
    assert type(encoder) is RecursiveNNSupervisedEncoder, type(encoder)
    acc = encoder.prediction_accuracy(sys.argv[2])
    print("Supervised Accuracy: %s%%" % (acc * 100))
