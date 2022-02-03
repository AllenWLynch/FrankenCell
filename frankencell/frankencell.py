import anndata
from scipy.stats import multivariate_hypergeom, lognorm
from tqdm import tqdm
import pandas as pd
from scipy import sparse
from functools import partial
import numpy as np
from joblib.parallel import Parallel, delayed


class CRP:
    
    def __init__(self, gamma, max_depth, max_width = 2):
        self.gamma = gamma
        self.max_depth = max_depth
        self.max_width = max_width
        self.tables = []
        self.n = 0
    
    def _nested_sum(self, table):
        total = 0
        if not isinstance(table, list):
            return table
        for i in table:
            if isinstance(i, list):
                total += self._nested_sum(i)
            else:
                total += i
        return total
        
    def _seat(self, table, depth):
        
        is_leaf= depth+1 >= self.max_depth
               
        n_seated = np.array([self._nested_sum(t) for t in table])
        n_total = n_seated.sum() + 1
        
        if n_total == 0:
            selected_table = 0
        else:
            gamma = self.gamma
            if len(table) == self.max_width:
                gamma = 0
                
            denom = gamma + n_total -1
            p_table = np.concatenate([n_seated/denom, [gamma/denom]], axis = -1)
                        
            selected_table = np.random.choice(range(len(table)+1), p = p_table)
        
        recurse_table = []
        if selected_table == len(table):
                        
            if is_leaf:
                table.append(1)
                return [selected_table]
            else:
                table.append(recurse_table)
                return [selected_table, *self._seat(recurse_table, depth+1)]
                     
        else:
            recurse_table = table[selected_table]
        
            if is_leaf:
                return [selected_table]
            else:
                return [selected_table, *self._seat(recurse_table, depth+1)]
    
    def new_customer(self):
        return [0,*self._seat(self.tables, 1)]
    
    
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
    
def generate_branching_process(
    branch_times,
    n_cells = 1000,
    gamma = 0.1,
    max_depth = 2,
    max_width = 2,
    ptime_alpha = 1.,
    ptime_beta = 1.,
    sigmoid_approach = True,
    sigmoid_aggression = 5
):

    assert(len(branch_times) == (max_depth +1))
    branch_times = np.array(branch_times)
    crp = CRP(gamma, max_depth, max_width=max_width)
    
    min_sigmoid, max_sigmoid = sigmoid(-0.5 * sigmoid_aggression), sigmoid(0.5 * sigmoid_aggression)
    
    cells = []
    for cell in range(n_cells):
        
        pseudotime = np.random.beta(ptime_alpha, ptime_beta)
        path = crp.new_customer()
                        
        level = np.argmin(pseudotime > branch_times) - 1
        progress = (pseudotime - branch_times[level])/(branch_times[level+1] - branch_times[level])
        
        if sigmoid_approach:
            x = sigmoid_aggression*(progress - 0.5)
            progress = (sigmoid(x) - min_sigmoid)/(max_sigmoid - min_sigmoid)
            
        mixing = np.concatenate([np.zeros(level), 
                                 [1-progress, progress],
                                np.zeros(len(path) - level - 1)])
        
        cells.append((pseudotime, path, mixing))
        
    return list(
        map(np.array, list(zip(*cells)))
    )


def get_idx_from_path(path):
    idx = [1]
    for p in path[1:]:
        idx.append(2*idx[-1] + p)
        
    return idx


def prepare_adata(adata, cell_states, 
                  cell_state_col = 'leiden',
                 counts_layer = 'counts'):
    
    return {
        c : (adata[adata.obs[cell_state_col] == c].layers[counts_layer].tocsr(), 
             np.array(adata[adata.obs[cell_state_col] == c].layers[counts_layer].sum(-1)).reshape(-1))
        for c in cell_states
    }


def mix_cells(read_depth, mixing_weights, *cells):
        
    def geom_sample_sparse_array(arr, n_samples):

        subsampled_reads = multivariate_hypergeom(arr.data, n_samples).rvs()
        return sparse.csr_matrix((subsampled_reads.reshape(-1), arr.indices, arr.indptr), shape = arr.shape)
        
    mixed_cell = sparse.csr_matrix(sparse.vstack([
        geom_sample_sparse_array(feature_counts, int(read_depth * m))
        for feature_counts, m in zip(map(lambda x : x.tocsr().astype(int), cells), mixing_weights)
    ]).sum(0))
    
    return mixed_cell


def sample_proportions(*,
    state_counts_rna, state_counts_atac, mixing_weights,
    rna_read_depth, atac_read_depth):
    
    rna_params = state_counts_rna, rna_read_depth, mixing_weights
    atac_params = state_counts_atac, atac_read_depth, mixing_weights
    
    while True:
        
        valid_cells_rna = [X[1] >= int(rna_read_depth * m) 
            for X, m in zip(state_counts_rna.values(), mixing_weights)]

        valid_cells_atac = [X[1] >= int(atac_read_depth * m) 
        for X, m in zip(state_counts_atac.values(), mixing_weights)]

        valid_cells = list(
            map(lambda x : (x[0] * x[1]).astype(bool), zip(valid_cells_rna, valid_cells_atac))
        )

        num_valid_cells_per_condition = np.array(list(map(sum, valid_cells)))

        if (num_valid_cells_per_condition > 0).all():
            
            samples = [np.random.choice(num_valid) for num_valid in num_valid_cells_per_condition]

            cells_rna = [
                X[0][valid_mask][sample] 
                for X, valid_mask, sample in zip(state_counts_rna.values(), valid_cells, samples)
            ]

            cells_atac = [
                X[0][valid_mask][sample] 
                for X, valid_mask, sample in zip(state_counts_atac.values(), valid_cells, samples)
            ]

            frankencell = mix_cells(rna_read_depth, mixing_weights, *cells_rna), \
                mix_cells(atac_read_depth, mixing_weights, *cells_atac)

            break

        else:
            rna_read_depth*=np.random.beta(1, 0.3)
            atac_read_depth*=np.random.beta(1, 0.3)
    
    return frankencell


def get_simplex_layout(mixing_weights):
    
    a, b, c = mixing_weights[:,0], mixing_weights[:,1], mixing_weights[:,2]

    # translate the data to cartesian corrds
    x = 0.5 * ( 2.*b+c ) / ( a+b+c )
    y = 0.5*np.sqrt(3) * c / (a+b+c)
    
    return x,y


def make_franken_cells(
    seed = None,
    n_cells = 1000,
    gamma = 0.1,
    max_depth = 2,
    max_width = 2,
    ptime_alpha = 1.,
    ptime_beta = 1.,
    sigmoid_approach = True,
    sigmoid_aggression = 5,
    cell_state_col = 'leiden',
    rna_counts_layer = None,
    atac_counts_layer = None,
    generate_cells = True,
    n_jobs = 1,*,
    branch_times,
    state_compositions,
    pure_states,
    rna_read_depth_distribution,
    atac_read_depth_distribution,
    rna_adata,
    atac_adata,
):
    
    assert((rna_adata.obs_names == atac_adata.obs_names).all())
    assert((rna_adata.obs[cell_state_col] == atac_adata.obs[cell_state_col]).all())

    try:
        lognorm(*rna_read_depth_distribution).rvs()
        lognorm(*atac_read_depth_distribution).rvs()
    except TypeError:
        raise TypeError('Invalid parameters for scipy.stats.lognorm read distribution')
    
    state_compositions = np.array(state_compositions)

    np.random.seed(seed=seed)
    
    pseudotime, paths, transition_mixing = generate_branching_process(
        branch_times,
        gamma = gamma, max_depth = max_depth, max_width = max_width,
        ptime_alpha= ptime_alpha, ptime_beta=ptime_beta,
        sigmoid_approach=sigmoid_approach, sigmoid_aggression=sigmoid_aggression,
        n_cells = n_cells,
    )
    
    paths = np.array([[0, *get_idx_from_path(p)] for p in paths])
    transition_mixing = transition_mixing[:,:,np.newaxis]

    mixing_weights = (state_compositions[paths.astype(int)] * transition_mixing).sum(1)
    x,y = get_simplex_layout(mixing_weights)
    
    if not generate_cells:
        return dict(
            mixing_weights = mixing_weights, 
            pseudotime = pseudotime, 
            simplex_x = x,
            simplex_y = y
        )
    
    else:

        rna_read_depth = lognorm(*rna_read_depth_distribution).rvs(n_cells)
        atac_read_depth = lognorm(*atac_read_depth_distribution).rvs(n_cells)

        state_counts_rna = prepare_adata(rna_adata, pure_states, 
                  cell_state_col = cell_state_col, counts_layer = rna_counts_layer)
        
        state_counts_atac = prepare_adata(atac_adata, pure_states, 
                  cell_state_col = cell_state_col, counts_layer = atac_counts_layer)
        
        franken_function = partial(sample_proportions,
                state_counts_rna = state_counts_rna, state_counts_atac = state_counts_atac)
        
        if n_jobs > 1:
            frankencells = Parallel(n_jobs=n_jobs, verbose = 0, pre_dispatch='2*n_jobs')(
                delayed(franken_function)\
                    (mixing_weights = weights, rna_read_depth = rna_rd, atac_read_depth = atac_rd)
                    for weights, rna_rd, atac_rd in tqdm(
                            zip(mixing_weights, rna_read_depth, atac_read_depth), 
                            total = len(mixing_weights)
                    )
                )
        else:
            frankencells = [
                franken_function(mixing_weights = weights, rna_read_depth = rna_rd, atac_read_depth = atac_rd)
                for weights, rna_rd, atac_rd in tqdm(
                    zip(mixing_weights, rna_read_depth, atac_read_depth), 
                    total = len(mixing_weights))
            ]
            
        
        franken_rna, franken_atac = map(sparse.vstack, list(zip(*frankencells)))
        
        obs_df = pd.DataFrame(
            [pseudotime, *mixing_weights.T],
            index = ['pseudotime', *['mix_weight_' + str(i) for i in range(3)]],
            columns = np.arange(n_cells),
        ).T
        
        obsm = dict(
            simplex = np.hstack([x[:,np.newaxis],y[:, np.newaxis]]),
            mixing_weights = mixing_weights,
        )

        generation_data = dict(
            seed = seed,
            n_cells = n_cells,
            gamma = gamma,
            max_depth = max_depth,
            max_width = max_width,
            ptime_alpha = ptime_alpha,
            ptime_beta = ptime_beta,
            sigmoid_approach = sigmoid_approach,
            sigmoid_aggression = sigmoid_aggression,
            cell_state_col = cell_state_col,
            rna_counts_layer = rna_counts_layer,
            atac_counts_layer = atac_counts_layer,
            branch_times = branch_times,
            state_compositions = state_compositions,
            pure_states = pure_states,
            rna_read_depth_distribution = list(rna_read_depth_distribution),
            atac_read_depth_distribution = list(atac_read_depth_distribution),
        )
        
        rna_frankendata = anndata.AnnData(
            X = sparse.csr_matrix(franken_rna), obs = obs_df,
            var = pd.DataFrame(index = rna_adata.var_names.copy()),
            obsm = obsm, uns = dict(generation_params = generation_data)
        )
        
        atac_frankendata = anndata.AnnData(
            X = sparse.csr_matrix(franken_atac), obs = obs_df,
            var = pd.DataFrame(index = atac_adata.var_names.copy()),
            obsm = obsm, uns = dict(generation_params = generation_data)
        )
        
        return rna_frankendata, atac_frankendata