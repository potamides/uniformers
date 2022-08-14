from datasets.load import load_metric as _load_metric
from os.path import dirname, isdir, join

def load_metric(path, *args, **kwargs):
    if isdir(local := join(dirname(__file__), path)):
        return _load_metric(local, *args, **kwargs)
    return _load_metric(path, *args, **kwargs)
