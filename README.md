# iros-2022-cloth-manipulation

## Installation
Follow the installation from our [keypoint detection](https://github.com/tlpss/keypoint-detection) repo.
Activate the created env `conda activate keypoint-detection`.
Then run `conda install jupyter`
`conda install -c conda-forge setuptools==59.5.0` due to a bug in PyTorch, should be fixed when upgrading.


Then you should be able to run the scripts in this repo.

## Dirty fixes
`if self._fast_norm:` in `timm` gave errors so I commented those lines.