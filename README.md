# Quick Start
* Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) package manager.
* Create and activate conda environment.

```shell
conda env create -f environment.yml
conda activate cs236-dp
```

* Download dataset and baseline checkpoints and logs.

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

* Visualize baselines logs and your experiments using Tensorboard.

```shell
tensorboard --logdir out
```

> NOTE: Metrics logged during training (e.g. IS, FID, KID) are approximations computed using limited data. Use `eval.py` to compute accurate metrics.

# Baselines
The baseline models are Residual SNGANs from [Mimicry: Towards the Reproducibility of GAN Research](https://github.com/kwotsin/mimicry).

Resolution                |32x32                       |64x64
:------------------------:|:-------------------------:|:-------------------------:
Seed                      |0                          |0
Batch Size                |64                         |64
β<sub>1</sub>             |0                          |0
β<sub>2</sub>             |0.9                        |0.9
lr                        |2e-4                       |2e-4
lr<sub>idecay</sub>       |Linear                     |Linear
n<sub>iter</sub>          |150k                       |150k
n<sub>dis</sub>           |5                          |5
IS                        |6.212                      |7.234
FID                       |42.452                     |68.360
KID                       |0.02734                    |0.06240
Samples                   |![](https://user-images.githubusercontent.com/50810315/135712701-9a154614-1703-4aa4-94a3-54db05908dd8.png)   |![](https://user-images.githubusercontent.com/50810315/135712698-e7294a67-949b-482f-9212-075a7ddb59a6.png)

