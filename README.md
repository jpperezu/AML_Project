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
```

To be able to run the scripts clone the repository and do the following soft link in the path you cloned the repository:

``` bash
ln -s /media/SSD3/jpperezu/Efficient_DDPMs/ddpm_nas
```

To reproduce the images used in Figure 1 and 4 in the paper run:
``` bash
Fig_1_4.sh as bash Fig_1_4.sh
```

To reproduce Figure 2, Figure 3, Figure 5 run: 
``` bash
python Fig_2.py
python Fig_3.py
python Fig_5.py
```

To do test in the whole dataset used run:

``` bash
bash test.sh
```

To run the demo for a particular prompt run:

``` bash
bash main.sh
```
Choose the policy you want to use (make sure to adapt num_inference_steps to your policy)  Here i use #AG 0.9994 (32 NFEs)

