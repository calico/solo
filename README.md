# solo
### Why
Cells subjected to single cell RNA-seq have been through a lot, and they'd really just like to be alone now, please. If they cannot escape the cell social scene, you end up sequencing RNA from more than one cell to a barcode, creating a *doublet* when you expected single cell profiles.

**_solo_** is a neural network framework to classify doublets, so that you can remove them from your data and clean your single cell profile.

### Quick set up
Run the following to clone and set up ve.
`git clone git@github.com:calico/solo.git && cd solo && conda env create -n solo && conda activate solo && pip install -e .`

If you don't have conda follow the instructions here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/

### How
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

Data file can be h5ad or loom file format.

Usage: `demultiplex [options] <model_json> <data_file>`

For help do: `demultiplex -h`

model_json example:
```
{
  "priors": [0.01, 0.5, 0.49]
}
```

Data file can be h5ad or loom file format.
