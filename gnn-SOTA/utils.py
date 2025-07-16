import scanpy as sc
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from torch_geometric.utils import from_scipy_sparse_matrix

# This is from https://github.com/JinmiaoChenLab/GraphST/blob/main/GraphST/GraphST.py
def preprocess(adata, n_hvg):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_hvg)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    adata.obsm['hvg_feature'] = feat


def construct_interaction_KNN(adata: AnnData, n_neighbors: int = 3, store_key: str = 'adj'):
    """
    Constructs a symmetric KNN graph based on spatial coordinates.
    Stores the binary adjacency matrix in `adata.obsm[store_key]`.

    Parameters:
    - adata: AnnData object with .obsm['spatial'] coordinates
    - n_neighbors: Number of neighbors to connect each node to
    - store_key: Key under adata.obsm to store the adjacency matrix
    """
    if 'spatial' not in adata.obsm:
        raise ValueError("Missing `adata.obsm['spatial']` coordinates.")

    position = adata.obsm['spatial']
    n_spot = position.shape[0]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)

    x = np.repeat(np.arange(n_spot), n_neighbors)
    y = indices[:, 1:].flatten()

    interaction = np.zeros((n_spot, n_spot), dtype=np.float32)
    interaction[x, y] = 1

    # Make symmetric
    adj = interaction + interaction.T
    adj[adj > 1] = 1  # Avoid double edges

    adata.obsm[store_key] = adj
    print(f"Graph constructed with {n_neighbors} neighbors and stored in adata.obsm['{store_key}'].")

def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj_normalized.toarray()

def preprocess_adj(adj: np.ndarray) -> np.ndarray:
    """
    Normalize and add self-connections for GCN models.
    """
    adj_normalized = normalize_adj(adj)
    adj_normalized += np.eye(adj.shape[0])
    return adj_normalized

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
   """Convert a scipy sparse matrix to a torch sparse tensor."""
   sparse_mx = sparse_mx.tocoo().astype(np.float32)
   indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
   values = torch.from_numpy(sparse_mx.data)
   shape = torch.Size(sparse_mx.shape)
   return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)  

def dense_to_sparse_edge_index(adj: np.ndarray):
    sparse_adj = sp.coo_matrix(adj)
    edge_index, _ = from_scipy_sparse_matrix(sparse_adj)
    return edge_index
