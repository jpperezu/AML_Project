
import torch
import torch.nn as nn
from PIL import Image
import wandb
import json
from tqdm.auto import tqdm
from torch import autocast
import random
import numpy as np
import argparse
import csv
import os

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from diffusers import DPMSolverSinglestepScheduler
from diffusers import DPMSolverMultistepScheduler


parser = argparse.ArgumentParser()
parser.add_argument('--num_inference_steps', type=int, default=20, help='Num of inference steps @DPM solver.') #20
parser.add_argument('--policy', nargs='+', type=int, default=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], help='Hardcoded policy')
parser.add_argument('--seed', type=int, default=32, help='random seed')
parser.add_argument('--base_folder', type=str, default="GENERATE_IMAGES", help='Base folder name')
parser.add_argument('--json', type=str, default="Fig_1.json", help='json file name')


args = parser.parse_args()

#Create folder structure:

# Function to create folder if it doesn't exist
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder created: {path}")
    else:
        print(f"Folder already exists: {path}")

# Base folder name
base_folder = args.base_folder 

# Check and create the main folder
create_folder(base_folder)

# Create sub-folders "noise_seed" and "steps" inside 'regression_dataset'
sub_folders = ["images"]
for folder_name in sub_folders:
    folder_path = os.path.join(base_folder, folder_name)
    create_folder(folder_path)
  
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

scheduler = DPMSolverSinglestepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
# scheduler.solver_order = 3  # for unconditional sampling

vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device) 

height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = args.num_inference_steps            # Number of denoising steps
batch_size = 1
policy= np.array(args.policy)       # Hardcoded policy

#Define choose noise function for policy:
guidance_scale = [7.5] # Change for mixed guidance

def choose_noise(possible_choice, current_noise):  # 0: uncond, 1: text, 2: guidance
  
  noise_choice_idx = int(current_noise)
  possible_options = [
    possible_choice[0],
    possible_choice[1],
    possible_choice[0] + guidance_scale[0] * (possible_choice[1] - possible_choice[0])
    # possible_choice[0] + guidance_scale[1] * (possible_choice[1] - possible_choice[0]),
    # possible_choice[0] + guidance_scale[2] * (possible_choice[1] - possible_choice[0]),
    #possible_choice[0] + guidance_scale[3] * (possible_choice[1] - possible_choice[0]),
    #possible_choice[0] + guidance_scale[4] * (possible_choice[1] - possible_choice[0])
  ] #Uncomment for mixed guidance
  return possible_options[noise_choice_idx], str(noise_choice_idx)

# Open the JSON file
with open(args.json, 'r') as file:  
    data = json.load(file)
  
for filename, prompt in tqdm(data.items()):

  print(f'generating prompt: {prompt}')

  generator = torch.manual_seed(args.seed)   # Seed generator to create the inital latent noise
  text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

  with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

  max_length = text_input.input_ids.shape[-1]
  uncond_input = tokenizer(
      [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
  )
  with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]  

  text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

  latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
  )

  latents = latents.to(torch_device)

  scheduler.set_timesteps(num_inference_steps)
  latents = latents * scheduler.init_noise_sigma
  prev_noise_pred = torch.zeros_like(latents)

  for k, t in tqdm(enumerate(scheduler.timesteps)):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    
    # predict the noise residual
    with torch.no_grad():
      noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    
    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred, noise_type = choose_noise([noise_pred_uncond, noise_pred_text, prev_noise_pred], policy[k])
    prev_noise_pred = noise_pred.clone()

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

  # scale and decode the image latents with vae
  latents = 1 / 0.18215 * latents

  with torch.no_grad():
    image = vae.decode(latents).sample

  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  image = (image * 255).round().astype("uint8").squeeze()

  image = Image.fromarray(image)
  
  # Save images
  image.save(f"./{base_folder}/{sub_folders[0]}/{filename}_{args.seed}.png")
  print(f'image {filename} saved successfully.')




