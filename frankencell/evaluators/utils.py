import numpy as np

def make_label_array(values):
    values= values.astype(int)
    labels = np.zeros((len(values), int(values.max())+1))
    labels[np.arange(len(values)), values] = 1
    return labels


def jsonize_generation_params(params):

    def norm(v):
    
        if isinstance(v, (np.ndarray, list)):
            return np.array(v).tolist()
        elif isinstance(v, np.int64):
            return int(v)

    return {k : norm(v) for k, v in params.items()}