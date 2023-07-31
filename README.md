# AL-foundation-models
This code base serves as a highly performant and efficient implementation of modern active learning strategies, along with the introduction of **DropQuery**. This repository is also amenable to the use of open-source foundation models.

See the following publication for more **DropQuery** details:
> **Revisiting active learning queries in the era of Foundation Models** <br> [...] et al.<br> under review.

## Installation instructions
```
conda env create -f env.yml
conda activate alfm_env

git clone git@github.com:sanketx/AL-foundation-models.git
cd Al-foundation-models
conda develop .
```

Make a copy of `ALFM/.env.example` and name it `ALFM/.env`. Change the paths in it to reflect your local setup.

## Feature extraction
To extract features from a foundation model, you must first download the model weights and place them in `MODEL_CACHE_DIR` as specified in your `.env` file. Feature extraction sweeps are configured in `ALFM/conf/feature_extraction.yaml`.

Run `python -m ALFM.feature_extraction` to run the sweep. 

There are many options to configure the sweep, so specify command line overrides with `python -m ALFM.feature_extraction model=dino_vit_S14 dataset=cifar10,food101` to extract features for multiple models and datasets, for example. 

Features will be saved to `FEATURE_CACHE_DIR` as specified in your `.env` file. For multi-GPU inference, make the following changes
```
export SLURM_JOB_NAME=interactive  # otherwise pytorch lightning won't launch new processes
python -m ALFM.feature_extraction +trainer.strategy=ddp trainer.devices=4
```

## Active Learning
The following active learning strategies are implemented:

| Strategy | Paper | Previous code | Notes |
| --- | --- | --- | --- |
| Random, Uncertainty, Entropy, Margins | - | - | Random acquisition or querying points with highest predictive uncertainty, highest predictive entropy, or the smallest margin between top-2 predicted class probs, respectively. |
| BALD | [Gal et al.](https://arxiv.org/abs/1703.02910) | [BALD](https://github.com/lunayht/DBALwithImgData) | Bayesian active learning by disagreement; measure mutual information between model parameters and label entropy. |
| Power BALD | [Kirsch et al.](https://arxiv.org/pdf/2106.12059.pdf) | - | BALD acquisition but replacing top-k querying with random sampling. |
| Coreset | [Sener et al.](https://arxiv.org/abs/1708.00489) | [Coreset](https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py) | Query the points that maximize the minimum distance to the already queried points. |
| BADGE | [Ash et al.](https://arxiv.org/abs/1906.03671) | [BADGE](https://github.com/JordanAsh/badge) | Gradients inform uncertainty estimates; query the points with the highest gradient magnitude. |
| Alfa-Mix | [Parvaneh et al.](https://arxiv.org/abs/2203.07034) | [Alfa-Mix](https://github.com/AminParvaneh/alpha_mix_active_learning) | Query unlabeled points that when interpolated with labeled points, have high label inconsistency. |
| Typiclust | [Hacohen et al.](https://arxiv.org/abs/2202.02794) | [Typiclust](https://github.com/avihu111/TypiClust) | Query the points that are typical and representative of the unlabeled data. |
| ProbCover | [Yehuda et al.](https://arxiv.org/pdf/2205.11320.pdf) | [ProbCover](https://github.com/avihu111/TypiClust) | Like coreset acquisition, but mimics the max coverage problem by leveraging representation learning and seeing the coverage of $\delta$-balls around candidate points. |
| ***DropQuery*** (Ours) | - | [DropQuery](https://github.com/sanketx/AL-foundation-models) | A simple method where the inconsistency of predictions on features with dropout is a proxy for uncertainty. |

### Active Learning Experiments

Run `python -m ALFM.al_train` to run an Active Learning experiment. 

Default values are set in `ALFM/conf/al_training.yaml` but can be overriden. Specify command line overrides with `python -m ALFM.al_train model=dino_vit_g14` to run experiments with particular models or datasets.

Guaranteed to be super fast ;)

### Custom Experiments

You can also run custom experiments by creating new config files in `ALFM/conf/dataset` for new datasets and `ALFM/conf/query_strategy` for custom Active Learning strategies. 

You can then use `python -m ALFM.feature_extraction dataset=your_custom_dataset.yaml` to extract foundation model features and `python -m ALFM.al_train query_strategy=your_custom_query_strategy.yaml` to run your custom experiments.

## Citation
If you find this code useful, please cite our paper
```
@misc{ALFM,
  title={Revisiting active learning queries in the era of Foundation Models},
  author={... et al.},
  journal={under review NeurIPS 2023},
  year={2023}
}
``` 