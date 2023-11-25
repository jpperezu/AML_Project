import os
import csv
import sys
import json
import time
import glob
import wandb
import random
import numpy as np
from ddpm_nas.misc import utils
import logging
import argparse
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt 

import torch
import torch.utils
import torch.nn as nn
from torch import autocast
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from diffusers import DPMSolverSinglestepScheduler
from diffusers import DPMSolverMultistepScheduler

from ddpm_nas.lion import Lion
from ddpm_nas.network import NetworkTest 
from ddpm_nas.cc3m_1000_test_dataset import CC3MDataset

from dreamsim import dreamsim 



parser = argparse.ArgumentParser()
parser.add_argument('--test_data', type=str, default="test_1000_dataset", help='Load imgs.')
parser.add_argument('--folder_path', type=str, default="Fig_3", help='Save generated imgs.')
parser.add_argument('--test_json', type=str, default="Fig_3.json", help='JSON with imgs.') #test_100
parser.add_argument('--threshold', type=float, default=0.9990) #0.997
parser.add_argument('--num_inference_steps', type=int, default=20, help='Num of inference steps @DPM solver.')
parser.add_argument('--gamma', type=float, default=1e-3, help='Gamma regulates the alpha initialization values.')
parser.add_argument('--temperature', type=int, default=1, help='Temperature value that regulates the alpha weights.')
parser.add_argument('--max_cost', type=int, default=30, help='Max cost available to regulate the alphas.')
parser.add_argument('--lambda_cost', type=float, default=1.0, help='lambda to regulate the loss cost.')
parser.add_argument('--load_folder', type=str, default="debug-", help='Load model.')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--image_size', type=int, default=512, help='image size')
parser.add_argument('--optimizer', type=str, default='Lion', help='init learning rate')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--start_exp', type=str, default="00001", help='This is the index of the exp youre running. Define it outside this script.')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--wandb', type=bool, default=True, help='wandb')
parser.add_argument('--seed', type=int, default=32, help='random seed')

args = parser.parse_args()


percep_model, preprocess = dreamsim(pretrained=True)


def choose_noise(possible_choice, current_noise):  # 0: uncond, 1: text, 2: guidance
  noise_choice_idx = int(current_noise)
  
  return possible_choice[noise_choice_idx], str(noise_choice_idx)


def mse_calc(matrix1, matrix2):
  squared_diff = np.square(matrix1 - matrix2)
  mse = np.mean(squared_diff)

  return mse


def create_csv_from_string(mse, percep_dist, column_names, filename):
    input = [filename, mse, percep_dist]
    # input = input.split('_')

    exp = column_names[-1]
    
    file_path = f'{args.folder_path}/results.csv'
    file = False
    if os.path.exists(file_path):
        print("File exists")
        file = True

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file:
          # Write header row
          writer.writerow(column_names)
        
        # Write data rows
        writer.writerow(input)
        
    print('CSV file created successfully.')


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  else:
    # torch.cuda.empty_cache()
    torch.cuda.set_device(args.gpu)
    torch_device = "cuda"

  
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  # DDPM
  # 1. Load the autoencoder model which will be used to decode the latents into image space. 
  vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
  vae = utils.set_params(vae, False)

  # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
  tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
  text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
  text_encoder = utils.set_params(text_encoder, False)

  # 3. The UNet model for generating the latents.
  unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
  unet = utils.set_params(unet, False)

  scheduler = DPMSolverSinglestepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
  # scheduler.solver_order = 3  # for unconditional sampling

  # Inits
  vae = vae.to(torch_device)
  text_encoder = text_encoder.to(torch_device)
  unet = unet.to(torch_device) 

  height = args.image_size                        # default height of Stable Diffusion
  width = args.image_size                         # default width of Stable Diffusion
  assert height == width

  num_inference_steps = args.num_inference_steps       # Number of denoising steps

  guidance_scale = 7.5                # Scale for classifier-free guidance

  batch_size = args.batch_size

  # NAS model
  criterion = nn.L1Loss()
  criterion = criterion.to(torch_device)
  
  test_data = CC3MDataset(args.test_json, args.test_data, height)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=batch_size, pin_memory=True, num_workers=2)

  test_acc = test(test_queue, unet, tokenizer, text_encoder, scheduler, num_inference_steps, vae, criterion, batch_size, height, width, torch_device)
  logging.info('test_acc %f', test_acc)

def min_max_norm(noise_pred_uncond, noise_pred_text):
  noise_pred_uncond = (noise_pred_uncond - noise_pred_uncond.min())/ (noise_pred_uncond.max() - noise_pred_uncond.min())  
  noise_pred_text = (noise_pred_text - noise_pred_text.min())/ (noise_pred_text.max() - noise_pred_text.min())  
  return noise_pred_uncond, noise_pred_text


def norm_latents(vae, noise_pred_uncond, noise_pred_text):
  noise_pred_uncond = 1 / 0.18215 * noise_pred_uncond
  dec_noise_pred_uncond = vae.decode(noise_pred_uncond).sample
  image_uncond = (dec_noise_pred_uncond / 2 + 0.5).clamp(0, 1)
  # wandb.log({"unc": wandb.Image(image_uncond)})
  
  noise_pred_text = 1 / 0.18215 * noise_pred_text
  dec_noise_pred_text = vae.decode(noise_pred_text).sample
  image_text = (dec_noise_pred_text / 2 + 0.5).clamp(0, 1)
  # wandb.log({"cond": wandb.Image(image_text)})

  return dec_noise_pred_uncond, dec_noise_pred_text

def draw_choices(matrix):
  
  # Find the maximum value per row along with its index
  _, max_indices = torch.max(matrix, dim=1)
  fig, ax = utils.draw_choices(max_indices.numpy(), 'test')

def test(test_queue, unet, tokenizer, text_encoder, scheduler, num_inference_steps, vae, criterion, batch_size, height, width, torch_device):
  objs = utils.AvgrageMeter()
#   model.eval()
  min_val = 0 
  max_val = 1  
  fig, ax = plt.subplots(figsize=(15, 8))

  right_most_index = None
  all_sequences = []
  for step, (prompt, target, target_lat, seed, filename) in tqdm(enumerate(test_queue)):
    n = target.size(0)
    
    target = target.to(torch_device)
    target_lat = target_lat.to(torch_device)
    
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
 
    latents = seed.to(torch_device)

    latents = latents.to(torch_device)
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma

    cond_noises = []    
    distance_per_step, cosine_dist_per_step = [], []
    mean_cond, mean_unc = [], []
    print(prompt)
    
    for k, t in enumerate(scheduler.timesteps):
      latent_model_input = torch.cat([latents] * 2)
      
      latent_model_input = scheduler.scale_model_input(latent_model_input, t)  # AC: we don't need this for DPM solver

      # predict the noise residual
      with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
      
      # perform guidance
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
      
      noise_pred_uncond_aux = latent_model_input - noise_pred_uncond
      noise_pred_text_aux = latent_model_input - noise_pred_text

      # track distance of the noise_pred_uncond_aux and noise_pred_text_aux
      distance_per_step.append((criterion(noise_pred_uncond_aux, noise_pred_text_aux).cpu().numpy()))
      cosine_dist_per_step.append((F.cosine_similarity(noise_pred_uncond_aux.view(n,-1), noise_pred_text_aux.view(n,-1)).cpu().numpy()))

      latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Convert the list of arrays to a NumPy array
    data_array = np.array(cosine_dist_per_step)
    all_sequences.append(data_array)
    
    # Find indices where the value is greater than the threshold
    indices_greater_than_threshold = np.where(data_array > args.threshold)[0]

    # Find consecutive sequences of indices
    consecutive_sequences = []
    current_sequence = []

    for index in indices_greater_than_threshold:
        if not current_sequence or index == current_sequence[-1] + 1:
            current_sequence.append(index)
        else:
            current_sequence = [index]
    
    right_most_index_prompt = current_sequence[0]


    # Print or use consecutive_sequences as needed
    print("Consecutive sequences of indices:", consecutive_sequences)

    # Print or use right_most_index as needed
    print("Right-most index of the first consecutive sequence greater than the threshold:", right_most_index_prompt)

    # Plot the data with consecutive sequences highlighted
    plt.plot(data_array, label='Data')
    plt.scatter(indices_greater_than_threshold, data_array[indices_greater_than_threshold], color='red', label='Threshold Exceeding Points')

    if right_most_index is None or right_most_index_prompt > right_most_index:
      right_most_index = right_most_index_prompt
      # Append the last sequence
      consecutive_sequences.append(current_sequence)

  for sequence in consecutive_sequences:
      plt.axvspan(sequence[0], sequence[-1], facecolor='white', alpha=0.3, label='Consecutive Sequence')
  #plt.axhline(y=args.threshold, color='r', linestyle='--', label='Threshold')
  plt.ylim(args.threshold-0.001, 1)  # Customize the y-axis limits
  # plt.legend()
  plt.xlabel('Step index')
  plt.ylabel('Cosine similarity')
  #plt.title('Cosine similarity between uncond and cond noise per step')
  # plt.savefig(f"./{args.folder_path}/cosine_xt-cuc_per_step_{step}_cuttingprompts_{args.threshold}.png")
  # Print or use right_most_index as needed
  print("Right-most index of the dataset greater than the threshold:", right_most_index)
  
  all_seqs = np.concatenate(all_sequences, axis=-1)
  np.save(f'all_seqs_{args.num_inference_steps}DPM.npy', all_seqs)
  # plt.axhline(y=args.threshold, color='k', linestyle='--', label=r'AG $\bar{\gamma}=0.9990$')
  plt.axhline(y=0.9990, color='k', linestyle='--', label=r'AG $\bar{\gamma}=0.9990$')
  plt.axhline(y=0.9994, color='k', linestyle='--', label=r'AG $\bar{\gamma}=0.9994$')
  plt.axhline(y=0.9996, color='k', linestyle='--', label=r'AG $\bar{\gamma}=0.9996$')

  mean_all_seqs_2 = np.mean(all_seqs, axis=-1)
  plt.plot(mean_all_seqs_2, label='Mean', linewidth=6)
      
    # Check if the folder already exists
  if not os.path.exists(args.folder_path):
      # Create the folder
      os.makedirs(args.folder_path)
      
  plt.savefig(f"./{args.folder_path}/cosine_xt-cuc_per_step_{step}_cc3m_{args.num_inference_steps}DPM_{args.threshold}.png")

  breakpoint()

  if step  % 50 == 0:
    print('saving plot')
    # ax.set_ylim(0.0, 0.00008)  # Customize the y-axis limits
    ax.set_xlabel('solver Step')
    ax.set_ylabel('cosine sim.')
    plt.savefig(f"./{args.folder_path}/cosine_xt-cuc_per_step_{step}.png")
    # plt.savefig(f"./{args.folder_path}/scatter_mean_minmaxnorm_u-c_{step}.png")
  # if step == 100:
  #   break
  return objs.avg

if __name__ == '__main__':
  main() 
