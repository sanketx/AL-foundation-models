# AL-foundation-models
Active Learning in the era of Foundation Models

### Installation instructions
```
conda env create -f env.yml
conda activate alfm_env

git clone git@github.com:sanketx/AL-foundation-models.git
cd Al-foundation-models
conda develop .
```

Make a copy of `ALFM/.env.example` and name it `ALFM/.env`. Change the paths in it to reflect your local setup

### Feature extraction
Feature extraction sweeps are configured with `ALFM/conf/feature_extraction.yaml`.
Run `python -m ALFM.feature_extraction` to run the sweep. 
Specify command line overrides with `python -m ALFM.feature_extraction model=dino_vit_S14 dataset=cifar10`.
Features will be saved to `FEATURE_CACHE_DIR` as specified in your `.env` file.