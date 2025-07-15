# Introduction
This code repository is for benchmarking of Graph Neural Networks (GNNs) in spatial domain classification.
## Prepare
Make sure the data is downloaded from: [10x-Visium-preprocessed](https://drive.google.com/drive/folders/1BITQuzSGme2mEDXMgkY7zxes8S6Yrp7x?usp=sharing) 

The code to preprocess 10x-Visium data is `gnn-st/main_feature_sel_hvg.py`. The original dataset is available at: [10x-Visium-original](https://figshare.com/articles/dataset/10x_visium_datasets/22548901)

# Code usage
## Hyperparameter optimization for optimal GNNs architecture
```
python main.py --adata_dir [path_to_data] --model [MODEL_CLASS]
```
## Example notebook
The jupyter notebook in `gnn-SOTA` is an example to trained optimized GNN models. Modify the path to `best_model_config.json` to the evaluated model.
