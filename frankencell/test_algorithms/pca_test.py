from cmath import log
import scanpy as sc
import numpy as np
import subprocess
import anndata
import json
import os
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

def eigengap(eigvals):
    return np.max([np.argmax(eigvals[:-1] - eigvals[1:]) + 1, 3])

def preprocess_rna_data(rna_frankendata, min_cells = 30):

    rna_frankendata.layers['counts'] = rna_frankendata.X.copy()
    
    sc.pp.filter_genes(rna_frankendata, min_cells = min_cells)
    sc.pp.calculate_qc_metrics(rna_frankendata, inplace=True, log1p=False)
    sc.pp.normalize_total(rna_frankendata, target_sum=1e4)
    sc.pp.log1p(rna_frankendata)
    sc.pp.highly_variable_genes(rna_frankendata)


def preprocess_atac_data(atac_frankendata, min_cells = 30):
    sc.pp.calculate_qc_metrics(atac_frankendata, inplace=True, log1p=False)
    sc.pp.filter_genes(atac_frankendata, min_cells = min_cells)


def run_pca(rna_frankendata):

    sc.pp.regress_out(rna_frankendata, ['total_counts'])
    sc.pp.scale(rna_frankendata)
    sc.tl.pca(rna_frankendata)

    return eigengap(rna_frankendata.uns['pca']['variance'])


def run_lsi(atac_frankendata):

    features = TfidfTransformer().fit_transform(atac_frankendata.X)
    svd = TruncatedSVD(n_components=8)
    dimreduced = svd.fit_transform(features)
    
    atac_frankendata.obsm['X_lsi'] = StandardScaler().fit_transform(dimreduced)
    atac_frankendata.uns['svd'] = svd.singular_values_

    return eigengap(svd.singular_values_)


def get_slingshot_data(rna_frankendata, embedding, num_components, resolution):

    sc.pp.neighbors(rna_frankendata, use_rep=embedding, n_pcs = num_components)
    sc.tl.leiden(rna_frankendata, resolution=resolution)
    
    start_cluster = rna_frankendata.obs.iloc[rna_frankendata.obs.pseudotime.argmin()].leiden
    end_clusters = rna_frankendata.obs.iloc[[rna_frankendata.obs.mix_weight_1.argmax(), rna_frankendata.obs.mix_weight_2.argmax()]].leiden.values.astype(str)

    slingshot_data = dict(
        coordinates = rna_frankendata.obsm[embedding][:, :num_components].tolist(),
        clusters = rna_frankendata.obs.leiden.values.astype(str).tolist(),
        start_cluster = int(start_cluster),
        end_clusters = end_clusters.tolist(),
    )

    return slingshot_data


def run_slingshot(slingdata, out_prefix, resolution, Rscript_path):

    input_file = out_prefix + '_slingdata_resolution_{}.json'.format(resolution)
    output_file = out_prefix + '_slingresults_resolution_{}.csv'.format(resolution)

    with open(input_file, 'w') as f:
        json.dump(slingdata, f)

    script = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'test_algorithms', 'slingshot.R')
    subprocess.run(
        [Rscript_path, script, input_file, output_file],check=True
    )

    results = pd.read_csv(output_file)
    os.remove(input_file)
    os.remove(output_file)
    return results

def main(
        min_cells = 30,*,
        data, 
        resolutions,
        out_prefix,
        Rscript_path,
        style= 'pca',
    ):

    frankendata = anndata.read_h5ad(data)
    seed = int(frankendata.uns['generation_params']['seed'])

    np.random.seed(seed)
    random_order = np.random.permutation(len(frankendata))

    rna_frankendata, atac_frankendata = frankendata[random_order, frankendata.var.feature_type == "RNA"], \
        frankendata[random_order, frankendata.var.feature_type == "ATAC"]

    if style == 'pca':
        embedding = 'X_pca'
        preprocess_rna_data(rna_frankendata, min_cells = min_cells)
        num_components = run_pca(rna_frankendata)

    elif style == 'lsi':
        embedding = 'X_lsi'
        preprocess_atac_data(atac_frankendata, min_cells = min_cells)
        num_components = run_lsi(atac_frankendata)
        rna_frankendata.obsm[embedding] = atac_frankendata.obsm[embedding]
        rna_frankendata.uns['svd'] = atac_frankendata.uns['svd']

    for resolution in resolutions:
        print('Running test with resolution: ', resolution)
        results = run_slingshot(
            get_slingshot_data(rna_frankendata, embedding, num_components, resolution), 
            out_prefix, resolution, Rscript_path
        )

        rna_frankendata.obsm['slingresults_' + str(resolution)] = results.values[:,1:]

    return rna_frankendata


def get_parser():

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--data', '-d', type = str, required=True)
    parser.add_argument('--out-prefix', '-o', type = str, required=True)
    parser.add_argument('--style','-y', type = str, default = 'pca')
    parser.add_argument('--resolutions', '-r', type= float, nargs='+')
    parser.add_argument('--Rscript-path','-R', type = str, required=True)

    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    out_data = main(
        data = args.data,
        out_prefix = args.out_prefix,
        style = args.style,
        resolutions = args.resolutions,
        Rscript_path = args.Rscript_path
    )

    out_data.write_h5ad(args.out_prefix + '_results_adata.h5ad')