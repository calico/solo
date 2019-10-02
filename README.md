# solo
### Why
Cells subjected to single cell RNA-seq have been through a lot, and they'd really just like to be alone now, please. If they cannot escape the cell social scene, you end up sequencing RNA from more than one cell to a barcode, creating a *doublet* when you expected single cell profiles.

**_solo_** is a neural network framework to classify doublets, so that you can remove them from your data and clean your single cell profile.

### Quick set up
Run the following to clone and set up ve.
`git clone git@github.com:calico/solo.git && cd solo && conda env create -n solo && conda activate solo && pip install -e .`

If you don't have conda follow the instructions here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/

### How to computationally identify doublets

Usage: `solo [options] <model_json> <data_file>`

For help do: `solo -h`

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
   output directory structure
 Markup : *  `is_doublet.npy`   
          * `vae.pt`
          *  `classifier.pt`
          * `latent.npy`              
          * `preds.npy`
          * `scores.npy`	
          * `real_cells_dist.pdf`  
          *  `accuracy.pdf` 
          *  `train_v_test_dist.pdf`
          *  `roc.pdf`	
          *  `scores_sim.npy`
          *  `preds_sim.npy`	
          *  `is_doublet_sim.npy` 
Data file can be h5ad or loom file format.

### How to identify demultiplex cell hashing data

Usage: `demultiplex [options] <model_json> <data_file>`

For help do: `demultiplex -h`

model_json example:
```
{
  "priors": [0.01, 0.5, 0.49]
}
```

Outputs: 
  output_dir solo_demultiplex/
      hashing_demultiplexed.h5ad
      hashing_qc_plots.png
Data file can be h5ad.
