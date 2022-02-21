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


def get_mutual_information(x,y):

    x_marg = x.mean(0,keepdims = True)
    y_marg = y.mean(0, keepdims = True)

    joint = (x[:, np.newaxis, :] * y[:,:, np.newaxis])
    marg = (x_marg * y_marg.T)[np.newaxis, :,:]

    mutual_information = np.sum(joint*np.log2(joint/marg), axis = (-2,-1)).mean()

    return mutual_information

def main(data, results):

    data.obs = data.obs.join(results, how = 'left', rsuffix='_observed', lsuffix='_expected')

    pseudotime_concordance = spearmanr(data.obs.pseudotime_observed.values, data.obs.pseudotime_expected.values)[0]

    f1 = []
    for state_prediction in data.obs.columns[data.obs.columns.str.startswith('state_prediction')]:

        try:
            f1.append(
                f1_score(
                    make_label_array(data.obs.state.values), 
                    make_label_array(data.obs[state_prediction].values),
                    average = 'micro'
                )
            )
        except ValueError:
            pass
        
    f1 = max(f1)

    y = data.obs[['mix_weight_' + str(i) for i in range(3)]].values
    rna_topics = data.obsm['X_topic_compositions']
    atac_topics = data.obsm['ATAC_topic_compositions']
    both_topics = np.hstack([rna_topics,atac_topics])/2

    rna_mutual_information = get_mutual_information(y, rna_topics)
    atac_mutual_information = get_mutual_information(y, atac_topics)
    both_mutual_information = get_mutual_information(y, both_topics)

    return dict(
        pseudotime_concordance = pseudotime_concordance,
        lineage_f1 = f1,
        rna_mutual_information = rna_mutual_information,
        atac_mutual_information = atac_mutual_information,
        both_mutual_information = both_mutual_information,
        generation_params = jsonize_generation_params(data.uns['generation_params']),
    )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data','-d',required=True, type = str)
    parser.add_argument('--test-results', '-r', required=True,type = str)
    parser.add_argument('--out-prefix', '-o', required=True, type = str)    
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    data = anndata.read_h5ad(args.data)
    results = pd.read_csv(args.test_results, sep = '\t').set_index('Unnamed: 0')
    results.index = results.index.astype(str)

    results = main(data, results)

    with open(args.out_prefix + '_summary.json', 'w') as f:
        print(json.dumps(results), file = f)

    print('Success!')