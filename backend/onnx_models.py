from functools import lru_cache
from hashlib import sha256
from threading import Lock

import onnxruntime as ort


_load_lock = Lock()


def verified_session(path, expected_sha256):
    # Concurrent first requests must share the same large CPU session.
    with _load_lock:
        return _load(path, expected_sha256)


@lru_cache(maxsize=2)
def _load(path, expected_sha256):
    weights = path.read_bytes()
    if sha256(weights).hexdigest() != expected_sha256:
        raise ValueError(f"模型文件与登记版本不一致：{path.name}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    return ort.InferenceSession(weights, sess_options=options, providers=["CPUExecutionProvider"])
