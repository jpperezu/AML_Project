# AML_Project
AML 2024 Project: Contribution to Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models

For the setup, you can create a new conda environment from 

``` bash
conda env create -f environment.yml
```

Then activate the environment by 

``` bash
conda activate pytorch_env_updated
```

Make sure you have the following packages 

``` bash
pip install diffusers==0.11.1
pip install transformers scipy ftfy accelerate
pip install wandb
pip install dreamsim
pip install open_clip_torch

