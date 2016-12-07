import sys

import numpy as np
from matplotlib import pylab as plt
from sklearn.decomposition import PCA

from data.dataimport import import_data
from encoders.baseencoder import AbstractEncoder

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage <encoderPkl> <dataset.json.gz>")
        sys.exit(-1)

    data = import_data(sys.argv[2])
    encoder = AbstractEncoder.load(sys.argv[1])

    query_expressions_with_annotations = {
        'a-b': 'r', 'b-a': 'b',
        'a-c': 'r', 'c-a': 'b',
        'a-(b+c)': 'r', '(b-a)+c': 'b',
        '(a+b)-(b+c)': 'r', '(b-b)-(a-c)': 'b',
        '(a+c)-(c+b)': 'r', '(c-c)-(a-b)': 'b',
        'a-(b-c)': 'r', 'b-(a+c)': 'b',
        'a-(c-b)': 'r', 'c-(a+b)': 'b',
    }
    pairs = [
        ('a-b', 'b-a'), ('a-c', 'c-a'),
        ('a-(b+c)', '(b-a)+c'),
        ('(a+b)-(b+c)', '(b-b)-(a-c)'),
        ('(a+c)-(c+b)', '(c-c)-(a-b)'),
        ('a-(b-c)', 'b-(a+c)'),
        ('a-(c-b)', 'c-(a+b)')
    ]

    used_expressions = {}


    def add_matching(expr_data):
        expr = ''.join(expr_data[0])
        if expr in query_expressions_with_annotations:
            used_expressions[expr] = expr_data


    for name, code in data.items():
        add_matching(code['original'])
        for noisy_sample in code['noise']:
            add_matching(noisy_sample)

    query_expressions = list(query_expressions_with_annotations)
    query_encodings = [encoder.get_encoding(used_expressions[e]) for e in query_expressions]
    query_encodings = np.array(query_encodings)

    pca = PCA(n_components=2)
    pca.fit(query_encodings)
    query_encodings = pca.transform(query_encodings)

    for exp1, exp2 in pairs:
        enc1 = query_encodings[query_expressions.index(exp1)]
        enc2 = query_encodings[query_expressions.index(exp2)]
        plt.plot([enc1[0], enc2[0]], [enc1[1], enc2[1]], '--', color=(.7, .7, .7))

    for i, expression in enumerate(query_expressions):
        color = query_expressions_with_annotations[expression]
        plt.scatter(query_encodings[i, 0], query_encodings[i, 1], color=color, s=20, marker='o')
        plt.annotate('$' + expression + '$', xy=(query_encodings[i, 0], query_encodings[i, 1]), color=color,
                     fontsize=15)

    plt.tick_params(
        which='both', bottom='off',
        top='off', left='off', right='off',
        labelbottom='off', labelleft='off')
    plt.tight_layout()
    plt.show()
