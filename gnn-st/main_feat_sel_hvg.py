import os
import scanpy as sc
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse
import pymrmr

# === Load dataset list ===
data_dir = 'dataset_10x-visium'
data_paths = [os.path.join(data_dir, x) for x in sorted(os.listdir(data_dir))]
data_names = [x.split('.')[0] for x in sorted(os.listdir(data_dir))]
print("Available files:", data_paths)
print("Dataset names:", data_names)

num_genes = 4096
HVG = []
for pid in tqdm(range(len(data_names))):
    adata = sc.read_h5ad(data_paths[pid])
    adata = adata[~adata.obs['Region'].isna()].copy()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=num_genes)
    selected_hvg = adata.var['highly_variable_rank'].dropna().index.to_numpy()
    HVG.append(selected_hvg)

# Convert HVG list to DataFrame with columns per PID
hvg_df = pd.DataFrame({data_names[i]: HVG[i] for i in range(len(data_names))})
# Save to CSV
hvg_df.to_csv('HVG_per_pid.csv', index=False)
# Find intersection (common HVGs across all pids)
common_hvg = set(HVG[0])
for hvg in HVG[1:]:
    common_hvg &= set(hvg)
# Convert to sorted list and save
common_hvg = sorted(list(common_hvg))
print(f"🔍 Common HVGs across all pids: {len(common_hvg)} genes")
# Save to file
pd.Series(common_hvg).to_csv('common_HVGs.csv', index=False, header=["gene"])

# Create output folders
os.makedirs(f'dataset_10x-visium_filtered_adatas_hvg-{num_genes}', exist_ok=True)
os.makedirs(f'dataset_10x-visium_adjacency_matrices_hvg-{num_genes}', exist_ok=True)

# Containers
adjacency_matrices = []
filtered_adatas = []

# Process each dataset
for pid in tqdm(range(len(data_names))):
    # Load and filter
    adata = sc.read_h5ad(data_paths[pid])
    adata = adata[~adata.obs['Region'].isna()].copy()

    # Filter genes by common HVG
    genes_intersection = np.intersect1d(adata.var_names, common_hvg)
    adata = adata[:, genes_intersection].copy()

    # Normalize and log
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Compute neighbors
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=10)

    # Get adjacency matrix
    A = adata.obsp['connectivities'].copy()

    # Save adjacency matrix
    adj_path = f'dataset_10x-visium_adjacency_matrices_hvg-{num_genes}/{data_names[pid]}.npz'
    scipy.sparse.save_npz(adj_path, A)

    # Save filtered AnnData (optional)
    filtered_path = f'dataset_10x-visium_filtered_adatas_hvg-{num_genes}/{data_names[pid]}.h5ad'
    adata.write(filtered_path)

    # Keep in memory
    adjacency_matrices.append(A)
    filtered_adatas.append(adata)