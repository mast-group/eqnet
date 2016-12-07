import hashlib


def file_md5(filename: str, blocksize: int = 2 ** 20) -> str:
    m = hashlib.md5()
    with open(filename, 'rb') as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()
