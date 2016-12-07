import gzip
import json
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage <fileToSplit> <testFileFromWhereKeysAreUsed>  <outputFile>")
        sys.exit(-1)

    with gzip.open(sys.argv[1], "rb") as f:
        file_to_split = json.loads(f.read().decode("utf-8"))

    with gzip.open(sys.argv[2], "rb") as f:
        test_file_to_use_the_keys_from = json.loads(f.read().decode("utf-8"))

    outputFile = {}
    failures = 0
    for snippet_name in test_file_to_use_the_keys_from:
        if snippet_name in file_to_split:
            outputFile[snippet_name] = file_to_split[snippet_name]
        else:
            print("Failed to find key %s" % snippet_name)
            failures += 1

    print("Failed to find %s keys" % failures)

    with gzip.GzipFile(sys.argv[3], 'wb') as outfile:
        outfile.write(bytes(json.dumps(outputFile), 'utf-8'))
