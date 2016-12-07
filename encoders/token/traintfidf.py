from encoders.token.tfidfencoder import TfidfEncoder

if __name__ == '__main__':
    import os, sys

    if len(sys.argv) != 2:
        print("Usage <trainFile>")
        sys.exit(-1)

    enc = TfidfEncoder(sys.argv[1])
    trained_file = os.path.basename(sys.argv[1])
    assert trained_file.endswith('-trainset.json.gz')
    pickled_filename = 'tfidfencoder-' + trained_file[:-len('-trainset.json.gz')] + '.pkl'
    enc.save(pickled_filename)
