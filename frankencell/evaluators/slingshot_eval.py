import numpy as np
import anndata
import pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
import argparse
import json

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

class TooManyLineages(ValueError):
    pass

def get_pseudotime_concordance(data, slingtrial):
    if slingtrial.shape[-1] != 2:
        raise TooManyLineages('Too many lineages')
        
    pseudotime = np.nanmean(slingtrial, axis = -1)
    
    return spearmanr(data.obs.pseudotime.values, pseudotime)[0]


def parse_states(slingtrial, flip = True):
    if slingtrial.shape[-1] != 2:
        raise TooManyLineages('Too many lineages')
        
    state_0 = np.isfinite(slingtrial).all(-1)
    
    state_1 = np.isfinite(slingtrial[:,int(flip)]) & ~state_0
    state_2 = np.isfinite(slingtrial[:,int(not flip)]) & ~state_0
    
    return np.hstack([x[:, np.newaxis] for x in [state_0, state_1, state_2]])


def get_lineage_f1(data, slingtrial, flip = True):
    
    return f1_score(
        make_label_array(data.obs.state.values),
        parse_states(slingtrial, flip=flip), average = 'macro'
    )


def get_scores(data):

    pseudotime, f1 = [],[]
    for slingtrial in [x for x in data.obsm.keys() if x[:5] == 'sling']:
        slingtrial= data.obsm[slingtrial].astype(float)

        try:
            pseudotime.append(get_pseudotime_concordance(data, slingtrial))
            f1.append(
                max(get_lineage_f1(data, slingtrial, flip = True), get_lineage_f1(data, slingtrial, flip = False))
            )
        except TooManyLineages:
            pass

    best_trial = np.argmax(f1)
    return pseudotime[best_trial], f1[best_trial]


def main(data):

    data = anndata.read_h5ad(data)

    pseudotime_concordance, f1 = get_scores(data)

    return dict(
        pseudotime_concordance = pseudotime_concordance,
        lineage_f1 = f1,
        generation_params = jsonize_generation_params(data.uns['generation_params']),
    )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data','-d',required=True, type = str)
    parser.add_argument('--out-prefix', '-o', required=True, type = str)    
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    results = main(args.data)

    with open(args.out_prefix + '_summary.json', 'w') as f:
        print(json.dumps(results), file = f)

    print('Success!')
