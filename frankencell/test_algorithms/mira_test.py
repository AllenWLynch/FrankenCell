import anndata
import scanpy as sc
import mira
import numpy as np
import argparse
import os
import pandas as pd

def preprocess_rna_data(rna_frankendata, min_cells = 30):

    rna_frankendata.layers['counts'] = rna_frankendata.X.copy()
    
    sc.pp.filter_genes(rna_frankendata, min_cells = min_cells)
    sc.pp.normalize_total(rna_frankendata, target_sum=1e4)
    sc.pp.log1p(rna_frankendata)
    sc.pp.highly_variable_genes(rna_frankendata)


def preprocess_atac_data(atac_frankendata, min_cells = 30):
    sc.pp.filter_genes(atac_frankendata, min_cells = min_cells)

def init_rna_model(seed):
    rna_model = mira.topics.ExpressionTopicModel(
        endogenous_key= 'highly_variable',
        exogenous_key= 'highly_variable',
        counts_layer='counts',
        beta=0.9,
        batch_size=64,
        seed = seed,
        encoder_dropout=0.015,
        num_topics = 4,
        kl_strategy='cyclic',
    )
    rna_model.set_learning_rates(0.0011661489989520215, 0.113232393998922)

    return rna_model
    

def init_atac_model(seed):

    atac_model = mira.topics.AccessibilityTopicModel(
        beta=0.93,
        batch_size=64,
        seed = seed,
        encoder_dropout=0.07,
        num_topics = 4,
        kl_strategy='cyclic',
    )
    atac_model.set_learning_rates(0.001, 0.1103771629085354)
    
    return atac_model


def tune_model(data, model, dtype,*,
    tuning_iters,
    train_size,
    batch_sizes,
    dropout_range,
    topic_range,
    out_prefix,
    cv,
):
    
    tuner = mira.topics.TopicModelTuner(
        model, cv = cv, 
        save_name = out_prefix + '_{}_study.pkl'.format(dtype),
        max_dropout = dropout_range[1],
        min_dropout = dropout_range[0],
        batch_sizes = batch_sizes,
        iters = tuning_iters,
        min_topics = topic_range[0],
        max_topics = topic_range[1],
    )

    tuner.train_test_split(data, train_size= train_size)
    tuner.tune(data)

    best_model = tuner.select_best_model(data)
    best_model.save(out_prefix + '_best_{}_model.pth'.format(dtype))

    return best_model


def run_pseudotime(data, embedding_key = 'X_joint_umap_features'):

    sc.pp.neighbors(data, use_rep=embedding_key, metric='manhattan',
        n_neighbors=15)

    mira.time.get_connected_components(data, key = 'connectivities')
    mira.time.get_transport_map(data, diffmap_distances_key='distances',
                            diffmap_coordinates_key= embedding_key,
                                start_cell = int(data.obs.pseudotime.argmin()))
    
    mira.time.get_branch_probabilities(data, 
                                   terminal_cells={'B' : int(data.obs.mix_weight_1.argmax()), 
                                                   'C' : int(data.obs.mix_weight_2.argmax())})

    state_to_label_map = { 'B, C' : 0, 'C, B' : 0, 'B' : 1, 'C' : 2 }
    state_predictions = []
    for threshold in np.geomspace(0.1, 10, 40):
        try:
            mira.time.get_tree_structure(data, threshold = threshold)
            state_predictions.append(
                data.obs.tree_states.map(state_to_label_map).copy().values
            )
        except ValueError:
            break

    return data.obs.mira_pseudotime.values, state_predictions


def format_results(data, pseudotime, state_predictions, out_prefix):

    results = pd.DataFrame(
        {
            'pseudotime' : pseudotime,
            **{
                'state_prediction_' + str(i) : state_predictions[i] 
                for i in range(len(state_predictions))
            }
        },
        index = data.obs_names.values,
    )

    results.to_csv(out_prefix + '_test-results.tsv', sep = '\t')


def main(*,
    data,
    out_prefix,
    tuning_iters = 32,
    batch_sizes = [32,64,128],
    topic_range = [3,8],
    dropout_range = [0.001, 0.1],
    cv = 5,
    train_size=0.8,
    min_cells_rna = 30,
    min_cells_atac = 30,
    embedding_key = 'X_joint_umap_features',
    retrain = True,
):

    frankendata = anndata.read_h5ad(data)
    seed = int(frankendata.uns['generation_params']['seed'])

    np.random.seed(seed)
    random_order = np.random.permutation(len(frankendata))

    rna_frankendata, atac_frankendata = frankendata[random_order, frankendata.var.feature_type == "RNA"], \
        frankendata[random_order, frankendata.var.feature_type == "ATAC"]

    preprocess_rna_data(rna_frankendata, min_cells = min_cells_rna)
    preprocess_atac_data(atac_frankendata, min_cells = min_cells_atac)

    training_args = dict(
        tuning_iters = tuning_iters, cv = cv, batch_sizes = batch_sizes,
        topic_range = topic_range, dropout_range = dropout_range, 
        train_size = train_size, out_prefix = out_prefix
    )

    rna_model, atac_model = init_rna_model(seed), init_atac_model(seed)

    rna_model_path = out_prefix + '_best_rna_model.pth'
    if not os.path.exists(rna_model_path) or retrain:
        rna_model = tune_model(rna_frankendata, rna_model, 'rna', **training_args)
        #rna_model.fit(rna_frankendata)
    else:
        rna_model = mira.topics.ExpressionTopicModel.load(rna_model_path)

    rna_model.predict(rna_frankendata)
    rna_model.get_umap_features(rna_frankendata, box_cox = 0.5)
    
    atac_model_path = out_prefix + '_best_atac_model.pth'
    if not os.path.exists(atac_model_path) or retrain:
        atac_model = tune_model(atac_frankendata, atac_model, 'atac', **training_args)
    else:
        atac_model = mira.topics.AccessibilityTopicModel(atac_model_path)
        
    atac_model.predict(atac_frankendata)
    atac_model.get_umap_features(atac_frankendata, box_cox = 0.5)

    rna_frankendata, atac_frankendata = mira.utils.make_joint_representation(
        rna_frankendata, atac_frankendata
    )

    rna_frankendata.obsm['ATAC_umap_features'] = atac_frankendata.obsm["X_umap_features"]
    rna_frankendata.obsm['ATAC_topic_compositions'] = atac_frankendata.obsm["X_topic_compositions"]

    pseudotime, state_predictions = run_pseudotime(rna_frankendata, embedding_key= embedding_key)

    format_results(rna_frankendata, pseudotime, state_predictions, out_prefix + '_' + embedding_key)

    return rna_frankendata


def get_parser():

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--data', '-d', type = str, required=True)
    parser.add_argument('--out-prefix', '-o', type = str, required=True)
    parser.add_argument('--embedding-key','-e', type = str, default='X_joint_umap_features')
    parser.add_argument('--skip-training', action = 'store_const', const = True,
        default = False)
    parser.add_argument('--tuning-iters','-i', default=32, type = int)
    parser.add_argument('--batch-sizes', '-b',type = int, nargs='+', default=[32,64,128])
    parser.add_argument('--topic-range','-t', type = int, nargs=2, default=[3,5])
    parser.add_argument('--dropout-range','-do', type = float, nargs = 2, default=[0.001, 0.1])
    parser.add_argument('--cv','-cv', type = int, default=5)
    parser.add_argument('--train-size', default=0.8, type = float)

    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    out_data = main(
        data = args.data,
        out_prefix = args.out_prefix,
        tuning_iters = args.tuning_iters,
        batch_sizes = args.batch_sizes,
        topic_range = args.topic_range,
        dropout_range = args.dropout_range,
        cv = args.cv,
        train_size= args.train_size,
        embedding_key= args.embedding_key,
        retrain=not args.skip_training,
    )

    out_data.write_h5ad(args.out_prefix + '_results_adata.h5ad')
