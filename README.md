# solo
### Why
Cells subjected to single cell RNA-seq have been through a lot, and they'd really just like to be alone now, please. If they cannot escape the cell social scene, you end up sequencing RNA from more than one cell to a barcode, creating a *doublet* when you expected single cell profiles.

**_solo_** is a neural network framework to classify doublets, so that you can remove them from your data and clean your single cell profile.

### Quick set up
Run the following to clone and set up ve.
`git clone git@github.com:calico/solo.git && cd solo && conda env create -n solo  python=3.6 && conda activate solo && pip install -e .`

If you don't have conda follow the instructions here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/

```Usage: solo [options] <model_json> <data_file>

Options:
  -h, --help            show this help message and exit
  -d DOUBLET_DEPTH      Depth multiplier for a doublet relative to the
                        average of its constituents [Default: 2.0]
  -g                    Run on GPU [Default: False]
  -o OUT_DIR            
  -r DOUBLET_RATIO      Ratio of doublets to true cells
                        [Default: 2.0]
  -s SEED               Seed VAE model parameters
  -k KNOWN_DOUBLETS     Experimentally defined doublets tsv file
  -t DOUBLET_TYPE       Please enter multinomial,
                        average, or sum [Default: multinomial]
  -e EXPECTED_NUMBER_OF_DOUBLETS
                        Experimentally expected number of doublets
```
                        
model_json example:
```
{
  "n_hidden": 384,
  "n_latent": 64,
  "n_layers": 1,
  "cl_hidden": 128,
  "cl_layers": 1,
  "dropout_rate": 0.2,
  "learning_rate": 0.001,
  "valid_pct": 0.10
}
```

Outputs:
* `is_doublet.npy`  np boolean array, true if a cell is a doublet, differs from `preds.npy` if `-e expected_number_of_doublets` parameter was used
* `vae.pt` scVI weights for vae
* `classifier.pt` scVI weights for classifier
* `latent.npy` latent embedding for each cell             
* `preds.npy` doublet predictions
* `scores.npy`	doublet scores
* `real_cells_dist.pdf` histogram of distribution of doublet scores
*  `accuracy.pdf` accuracy plot test vs train
*  `train_v_test_dist.pdf` doublet scores of test vs train
*  `roc.pdf`	roc of test vs train
*  `scores_sim.npy` see above but for simulated doublets
*  `preds_sim.npy`	see above but for simulated doublets
*  `is_doublet_sim.npy` see above but for simulated doublets


### How to identify demultiplex cell hashing data CLI

Demultiplexing takes as input an h5ad file with only hashing counts. Counts can be obtained from your fastqs by using kite. See tutorial here: https://github.com/pachterlab/kite

```Usage: demultiplex [options] <model_json> <cell_hashing_data_file>

Options:
  -h, --help  show this help message and exit
  -o OUT_DIR            
  -c CLUSTERING_DATA    
  -p PRE_EXISTING_CLUSTERS
  -q PLOT_NAME   
```

model_json example:
```
{
  "priors": [0.01, 0.5, 0.49]
}
```

Outputs: 
*  `hashing_demultiplexed.h5ad` anndata with demultiplexing information in .obs
*  `hashing_qc_plots.png` plots of probabilites for each cell


### How to identify demultiplex cell hashing data in line

```
>>> from solo import demultiplex
>>> import anndata
>>> cell_hashing_data = anndata.read("cell_hashing_counts.h5ad")
>>> demultiplex.demultiplex_cell_hashing(cell_hashing_data) 
>>> cell_hashing_data.obs.head()
                  most_likeli_hypothesis  cluster_feature  negative_hypothesis_probability  singlet_hypothesis_probability  doublet_hypothesis_probability         Classification
index                                                                                                                                                                            
CCTTTCTGTCCGAACC                       2                0                     1.203673e-16                        0.000002                        0.999998                Doublet
CTGATAGGTGACTCAT                       1                0                     1.370633e-09                        0.999920                        0.000080  BatchF-GTGTGACGTATT_x
AGCTCTCGTTGTCTTT                       1                0                     2.369380e-13                        0.996992                        0.003008  BatchE-GAGGCTGAGCTA_x
GTGCGGTAGCGATGAC                       1                0                     1.579405e-09                        0.999879                        0.000121  BatchB-ACATGTTACCGT_x
AAATGCCTCTAACCGA                       1                0                     1.867626e-13                        0.999707                        0.000293  BatchB-ACATGTTACCGT_x
>>> demultiplex.plot_qc_checks_cell_hashing(cell_hashing_data)
```


* `most_likeli_hypothesis` 0 == Negative, 1 == Singlet, 2 == Doublet
* `cluster_feature` how the cell hashing data was divided if specified or done automatically by giving a cell by genes anndata object to the `cluster_data` argument when calling `demultiplex_cell_hashing`
* `negative_hypothesis_probability`  
* `singlet_hypothesis_probability`  
* `doublet_hypothesis_probability`         
* `Classification` The sample of origin for the cell or whether it was a negative or doublet cell.
