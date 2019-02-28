# loner
software to detect doublets

Run the following to clone and set up ve.
`git clone git@github.com:calico/loner.git && cd loner && conda env create -f environment.yml`

If you don't have conda follow the instructions here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/

Usage: `./loner.py [options] <model_json> <data_file>`

For help do: `./loner.py -h`

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
