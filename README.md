# solo -- Doublet detection via semi-supervised deep learning
### Why
Cells subjected to single cell RNA-seq have been through a lot, and they'd really just like to be alone now, please. If they cannot escape the cell social scene, you end up sequencing RNA from more than one cell to a barcode, creating a *doublet* when you expected single cell profiles. https://www.cell.com/cell-systems/fulltext/S2405-4712(20)30195-2

**_solo_** is a neural network framework to classify doublets, so that you can remove them from your data and clean your single cell profile.

We benchmarked **_solo_** against other doublet detection tools such as DoubletFinder and Scrublet, and found that it consistently outperformed them in terms of average precision. Additionally, Solo performed much better on a more complex tissue, mouse kidney. 

### Quick set up
Run the following to clone and set up ve.
`git clone git@github.com:calico/solo.git && cd solo && conda create -n solo python=3.7 && conda activate solo && pip install -e .`

Or install via pip
`conda create -n solo python=3.7 && conda activate solo && pip install solo-sc`

If you don't have conda follow the instructions here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/

### How to solo
```
usage: solo [-h] -j MODEL_JSON_FILE -d DATA_PATH
            [--set-reproducible-seed REPRODUCIBLE_SEED]
            [--doublet-depth DOUBLET_DEPTH] [-g] [-a] [-o OUT_DIR]
            [-r DOUBLET_RATIO] [-s SEED] [-e EXPECTED_NUMBER_OF_DOUBLETS] [-p]
            [-recalibrate_scores] [--version]

optional arguments:
  -h, --help            show this help message and exit
  -j MODEL_JSON_FILE    json file to pass VAE parameters (default: None)
  -d DATA_PATH          path to h5ad, loom, or 10x mtx dir cell by genes
                        counts (default: None)
  --set-reproducible-seed REPRODUCIBLE_SEED
                        Reproducible seed, give an int to set seed (default:
                        None)
  --doublet-depth DOUBLET_DEPTH
                        Depth multiplier for a doublet relative to the average
                        of its constituents (default: 2.0)
  -g                    Run on GPU (default: True)
  -a                    output modified anndata object with solo scores Only
                        works for anndata (default: False)
  -o OUT_DIR
  -r DOUBLET_RATIO      Ratio of doublets to true cells (default: 2)
  -s SEED               Path to previous solo output directory. Seed VAE
                        models with previously trained solo model. Directory
                        structure is assumed to be the same as solo output
                        directory structure. should at least have a vae.pt a
                        pickled object of vae weights and a latent.npy an
                        np.ndarray of the latents of your cells. (default:
                        None)
  -e EXPECTED_NUMBER_OF_DOUBLETS
                        Experimentally expected number of doublets (default:
                        None)
  -p                    Plot outputs for solo (default: False)
  -recalibrate_scores   Recalibrate doublet scores (not recommended anymore)
                        (default: False)
  --version             Get version of solo-sc (default: False)
```

Warning: If you are going directly from cellranger 10x output you may want to manually inspect your data prior to running solo.

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
* `vae` scVI directory for vae
* `classifier.pt` scVI directory for classifier
* `latent.npy` latent embedding for each cell             
* `preds.npy` doublet predictions
* `softmax_scores.npy` updated softmax of doublet scores (see paper), same as `no_update_softmax_scores.npy` now
* `no_update_softmax_scores.npy` raw softmax of doublet scores

* `logit_scores.npy`	logit of doublet scores
* `real_cells_dist.pdf` histogram of distribution of doublet scores
*  `accuracy.pdf` accuracy plot test vs train
*  `train_v_test_dist.pdf` doublet scores of test vs train
*  `roc.pdf`	roc of test vs train
*  `softmax_scores_sim.npy` see above but for simulated doublets
*  `logit_scores_sim.npy` see above but for simulated doublets
*  `preds_sim.npy`	see above but for simulated doublets
*  `is_doublet_sim.npy` see above but for simulated doublets


### How to demultiplex cell hashing data using HashSolo CLI

Demultiplexing takes as input an h5ad file with only hashing counts. Counts can be obtained from your fastqs by using kite. See tutorial here: https://github.com/pachterlab/kite

```
usage: hashsolo [-h] [-j MODEL_JSON_FILE] [-o OUT_DIR] [-c CLUSTERING_DATA]
                [-p PRE_EXISTING_CLUSTERS] [-q PLOT_NAME]
                [-n NUMBER_OF_NOISE_BARCODES]
                data_file

positional arguments:
  data_file             h5ad file containing cell hashing counts

optional arguments:
  -h, --help            show this help message and exit
  -j MODEL_JSON_FILE    json file to pass optional arguments (default: None)
  -o OUT_DIR            Output directory for results (default:
                        hashsolo_output)
  -c CLUSTERING_DATA    h5ad file with count transcriptional data to perform
                        clustering on (default: None)
  -p PRE_EXISTING_CLUSTERS
                        column in cell_hashing_data_file.obs to specifying
                        different cell types or clusters (default: None)
  -q PLOT_NAME          name of plot to output (default: hashing_qc_plots.pdf)
  -n NUMBER_OF_NOISE_BARCODES
                        Number of barcodes to use to create noise distribution
                        (default: None)
```

model_json example:
```
{
  "priors": [0.01, 0.5, 0.49]
}
```

Priors is a list of the probability of the three hypotheses, negative, singlet,
or doublet that we test when demultiplexing cell hashing data. A negative cell's barcodes
doesn't have enough signal to identify its sample of origin. A singlet has
enough signal from single hashing barcode to associate the cell with ins
originating sample. A doublet is a cell barcode which has signal for more than one hashing barcode.
Depending on how you processed your cell hashing matrix before hand you may
want to set different priors. Under the assumption that you have subset your cell
barcodes using typical QC on your cell by genes matrix, e.g. min UMI counts,
percent mitochondrial reads, etc. We found the above setting of prior performed
well (see paper). If you have only done relatively light QC in transcriptome space
 I'd suggest an even prior, e.g. `[1./3., 1./3., 1./3.]`.


Outputs:
*  `hashsoloed.h5ad` anndata with demultiplexing information in .obs
*  `hashing_qc_plots.png` plots of probabilites for each cell


### How to demultiplex cell hashing data using HashSolo in line

```
>>> from solo import hashsolo
>>> import anndata
>>> cell_hashing_data = anndata.read("cell_hashing_counts.h5ad")
>>> hashsolo.hashsolo(cell_hashing_data)
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
