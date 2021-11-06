# Quick Start
* Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) package manager.
* Create and activate conda environment.

```shell
conda env create -f environment.yml
conda activate cs236-dp
```

> NOTE: PyTorch dependency specified in `environment.yml` uses CUDA 11.1. If CUDA 11.1 is unsupported on your environment, please install PyTorch separately by following the [official instructions](https://pytorch.org).

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

* Create leaderboard submission `submission.pth` (upload to Gradescope).

```shell
python eval.py --ckpt_path PATH_TO_CKPT --im_size RESOLUTION --submit
```

* Visualize baseline logs and your experiments using Tensorboard.

```shell
tensorboard --logdir out --bind_all
```

> NOTE: Metrics logged during training (e.g. IS, FID, KID) are approximations computed using limited data. Use `eval.py` to compute accurate metrics.

# Baseline Models
The baseline models are Residual SNGANs from [Mimicry: Towards the Reproducibility of GAN Research](https://github.com/kwotsin/mimicry).

    
Baselines                 |Baseline-32-150k           |Baseline-64-150k           |Baseline-32-295k           |Baseline-64-295k
:------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
Resolution                |32x32                      |64x64                      |32x32                      |64x64
Seed                      |0                          |0                          |236                        |236
Batch Size                |64                         |64                         |64                         |64
n<sub>iter</sub>          |150k                       |150k                       |295k                       |295k
n<sub>dis</sub>           |5                          |5                          |5                          |5
β<sub>1</sub>             |0                          |0                          |0                          |0
β<sub>2</sub>             |0.9                        |0.9                        |0.9                        |0.9
lr                        |2e-4                       |2e-4                       |2e-4                       |2e-4
lr<sub>decay</sub>        |Linear                     |Linear                     |Linear                     |Linear
IS                        |6.212                      |7.234                      |6.326                      |7.330
FID                       |42.452                     |68.360                     |35.339                     |62.250
KID                       |0.02734                    |0.06240                    |0.01984                    |0.05556
Samples                   |![](https://user-images.githubusercontent.com/50810315/135712701-9a154614-1703-4aa4-94a3-54db05908dd8.png)|![](https://user-images.githubusercontent.com/50810315/135712698-e7294a67-949b-482f-9212-075a7ddb59a6.png)|![](https://user-images.githubusercontent.com/50810315/135767248-06df651c-1bba-4f51-9d8c-31d3c9c9c4ff.png)|![](https://user-images.githubusercontent.com/50810315/135767245-e37ed07f-f71c-4a82-81e7-9b88277f73aa.png)

> NOTE: Use 150k baselines to benchmark models trained from scratch and use 295k baselines if your models don't involve training from scratch.
