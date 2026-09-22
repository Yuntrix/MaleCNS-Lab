# Scientific data required separately

The sharing copy retains the original source paths but does not bundle binary datasets, cached arrays or generated scientific reports. `DATA_MANIFEST.csv` lists the original scientific input/cache filenames and sizes, without personal machine paths. It is an inventory, not a download manifest.

## Core simulation

The default LIF engines expect these compatible files together:

- `data/processed/lif-connectome-csr.npz`
- `data/processed/simulation-neurons.parquet`

The OBS overlay and visual/laterality experiments additionally reference:

- `data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `data/processed/simulation-edges.parquet` for several retina-building scripts.
- Retina spatial maps, laterality indices and other arrays listed in the manifest, depending on the selected script.

Some scripts can generate particular index/map caches, but a complete raw-to-simulation preprocessing pipeline was not established in this review. Do not assume that copying raw data alone recreates the exact processed matrices. Neuron indices, row ordering and matrix orientation must agree across the files. Ask the maintainer for the matching data release and preparation instructions.

The raw data source is the [official MaleCNS download page](https://male-cns.janelia.org/download/). It lists the matching v1.0 annotation, neurotransmitter and connection-weight filenames. Raw data attribution belongs to FlyEM/HHMI Janelia, the University of Cambridge, MRC LMB and Google Research. The official project labels the dataset CC-BY. The maintainer still needs to document and distribute the compatible project-specific processed arrays and the exact processing recipe. The large raw connectome alone is approximately 1.05 GB. Do not substitute fabricated sample data and treat it as the original connectome.

Place the matching files at their original paths for local use. The `.gitignore` intentionally prevents accidental commits of datasets and generated outputs. This source-only distribution supports review and collaboration; full simulation reproduction remains a documented setup requirement.
