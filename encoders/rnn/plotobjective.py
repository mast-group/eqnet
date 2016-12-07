import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from encoders.rnn.siameseencoder import RecursiveNNSiameseEncoder

if __name__ == '__main__':
    import sys
    import os

    if len(sys.argv) != 2:
        print("Usage <FileName>")
        sys.exit(-1)
    hyperparameters = dict(log_learning_rate=-1,
                           rmsprop_rho=.98,
                           momentum=0.8,
                           minibatch_size=10,
                           memory_size=32,
                           grad_clip=10,
                           log_init_scale_embedding=-1,
                           dropout_rate=0,
                           dissimilar_margin=.1)

    dataset = sys.argv[1]
    dset_name = os.path.basename(dataset)
    assert dset_name.endswith('.json.gz')
    dset_name = dset_name[:-len('.json.gz')]

    data_dump_name = 'datadump-' + dset_name + '.pkl'
    if not os.path.exists(data_dump_name):
        all_params = dict(hyperparameters)
        lm = RecursiveNNSiameseEncoder(dataset, hyperparameters)
        X, Y, Z = lm.scan_objective(dataset, 'Add')
        with open(data_dump_name, 'wb') as f:
            pickle.dump((X, Y, Z), f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(data_dump_name, 'rb') as f:
            X, Y, Z = pickle.load(f)

    # plt.figure()
    # CS = plt.contour(X, Y, np.log(-Z + 1e-20), levels=[-20, -15, -10, -5, -2, -1, -0, 1, 2, 3, 4])
    # plt.clabel(CS, inline=1, fontsize=10)
    Z_norm = np.log(-Z + 1e-20)
    im = plt.imshow(-np.clip(Z, -2, 0), interpolation='gaussian', origin='lower',
                    cmap=cm.gray, extent=(0, 1, 0, 1))
    levels = [-2, -1, 0, 1, 2, 3, 3.95, 4, 4.025, 4.04, 5, 6, 7, 8]
    CS = plt.contour(Z_norm, levels,
                     origin='lower',
                     linewidths=2,
                     extent=(0, 1, 0, 1))

    plt.xlabel("add_weight")
    plt.ylabel("subtract_weight")
    plt.title('Objective Values')
    plt.grid()
    plt.show()
