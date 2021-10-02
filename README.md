# Quick Start
* Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) package manager.
* Create and activate conda environment.

```shell
conda env create -f environment.yml
conda activate cs236-dp
```

* Download dataset and baselines.

```shell
python download.py
```

* Start training using baseline model.

```shell
python train.py --name EXPERMENT_NAME
```

* Evaluate trained models.

```shell
python eval.py --ckpt_path PATH_TO_CKPT --im_size RESOLUTION
```

* Visualize baselines and your experiments using Tensorboard.

```shell
tensorboard --logdir out
```

> NOTE: Metrics logged during training (e.g. IS, FID, KID) are approximations computed using limited data. Use `eval.py` to compute accurate metrics.

# Baselines
The baseline model is based on the Residual SNGAN implementation from [Mimicry: Towards the Reproducibility of GAN Research](https://github.com/kwotsin/mimicry).
