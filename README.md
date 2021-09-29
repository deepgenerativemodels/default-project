# Quick Start
* Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) package manager.
* Create and activate conda environment.

```shell
conda env create -f environment.yml
conda activate cs236-dp
```

* Start training using baseline model.

```shell
python train.py --name test
```

* Visualize experiments using Tensorboard.

```shell
tensorboard --logdir out
```

# Acknowledgement
The baseline model architecture is based off of the Residual SNGAN implementation from [Mimicry: Towards the Reproducibility of GAN Research](https://github.com/kwotsin/mimicry).
