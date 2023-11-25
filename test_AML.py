import os
import csv
import sys
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
import open_clip

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from basicsr.metrics import calculate_psnr, calculate_ssim

parser = argparse.ArgumentParser()
parser.add_argument('--test_data', type=str, default="test_500_IOU_dataset", help='Load imgs.') #test_1000_dataset
parser.add_argument('--folder_path', type=str, default="test_SWISH_debug", help='Save generated imgs.') #Change to expname_sweep #exp_name
parser.add_argument('--exp_folder', type=str, default="im-s_512_CC3M_9000_v2_cost-relu_gumbel_softmax_perceptual_entropy_temp_multiguidance", help='Save generated imgs.') #Change to expname_sweep #exp_name
parser.add_argument('--test_json', type=str, default="test_1000.json", help='JSON with imgs.') #Change if wanting to change number of images in json
parser.add_argument('--num_inference_steps', type=int, default=15, help='Num of inference steps @DPM solver.') #20
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

percep_model, preprocess = dreamsim(pretrained=True) #Disstance metric
#guidance_scale = [7.5, 3.25, 6.0, 9.0, 15]  
# guidance_scale = [7.5, 3.75, 15] 
# guidance_scale = [7.5, 6, 9] 

guidance_scale = [7.5] # C, U , CFG 

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
  ]
  return possible_options[noise_choice_idx], str(noise_choice_idx)


def resize_tensor(image_tensor):
  # Define preprocessing function
  resized_tensor = torch.nn.functional.interpolate(image_tensor, size=224, mode='bilinear', align_corners=False)

  return resized_tensor


class ClipScore(nn.Module):
  def __init__(self, model_name_or_path, torch_device='cuda'):
    super().__init__()
    self.model, _, _ = open_clip.create_model_and_transforms(model_name_or_path, pretrained='laion2b_s34b_b88k')
    self.tokenizer = open_clip.get_tokenizer(model_name_or_path)
    self.model.to(torch_device)
    self.torch_device = torch_device

  def forward(self, image, prompt):
    
    text = self.tokenizer(prompt).to(self.torch_device)

    image_features = self.model.encode_image(resize_tensor(image))
    text_features = self.model.encode_text(text)
    text_features = text_features.to(image_features.device)

    clipscore = torch.nn.functional.cosine_similarity(image_features, text_features)

    return clipscore
  

def mse_calc(matrix1, matrix2): 
  squared_diff = np.square(matrix1 - matrix2)
  mse = np.mean(squared_diff)

  return mse


def create_csv_from_string(mse, percep_dist, clip_score, ssim_value, psnr_value, psnr_BasicSR, ssim_BasicSR, column_names, filename, complete_folder_path):
    input = [filename, mse, percep_dist, clip_score, ssim_value, psnr_value, psnr_BasicSR, ssim_BasicSR]
    # input = input.split('_')

    exp = column_names[-1]
    
    file_path = f'{complete_folder_path}/results.csv'
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
  load_folder_path = args.load_folder #args.load_folder
  files = [file for file in os.listdir(load_folder_path) if "alphas" in file]
  file = sorted(files)[-1] #Select last epoch alphas file. alphas_eX file, X being last epoch
  
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

  guidance_scale = [7.5, 3, 10, 25]                # In this case does nothing. Just to fill NetworkTest parameters which previoulsy used it to calculate noise_pred in def test.

  batch_size = args.batch_size
  
  clipscore = ClipScore(model_name_or_path='ViT-g-14')
  clipscore = clipscore.to(torch_device)

  # NAS model
  criterion = nn.MSELoss()
  criterion = criterion.to(torch_device)
  
  # Load alphas (CHECK load_alphas in network!)
  model = NetworkTest(guidance_scale[0], num_inference_steps, args.gamma, args.temperature, os.path.join(load_folder_path, file))
  policy = torch.argmax(model.alphas, dim=-1).numpy()
  
  # if len(policy)>15: #for cliploss exps with conver_log_alphas.py
  #   policy = np.delete(policy, 1, axis=0)
  
  test_data = CC3MDataset(args.test_json, args.test_data, height)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=batch_size, pin_memory=True, num_workers=2)
  
  test_acc = test(test_queue, policy, unet, tokenizer, text_encoder, scheduler, num_inference_steps, vae, clipscore, batch_size, height, width, torch_device)
  logging.info('test_acc %f', test_acc)



def test(test_queue, policy, unet, tokenizer, text_encoder, scheduler, num_inference_steps, vae, clipscore, batch_size, height, width, torch_device):
  objs = utils.AvgrageMeter()
  min_val = 0 
  max_val = 1 

  mse_values, percep_values, clip_values, ssim_values, psnr_values, psnr_BasicSR_values, ssim_BasicSR_values = [], [], [], [], [], [], []
  
  for step, (prompt, target, target_lat, seed, filename) in tqdm(enumerate(test_queue), desc='Testing images...'):
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
    
    prev_noise_pred = torch.zeros_like(latents)
    
    for k, t in enumerate(scheduler.timesteps):
      latent_model_input = torch.cat([latents] * 2)
      
      latent_model_input = scheduler.scale_model_input(latent_model_input, t)  # AC: we don't need this for DPM solver

      # predict the noise residual
      with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
      
      # perform guidance
      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    
      noise_pred, noise_type = choose_noise([noise_pred_uncond, noise_pred_text, prev_noise_pred], policy[k])
      cond_noises.append(noise_type)
      
      prev_noise_pred = noise_pred.clone()
      latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    target_lat = 1 / 0.18215 * target_lat

    # loss_latents = criterion(latents, target_lat)

    image = vae.decode(latents).sample
    image = (image / 2 + 0.5)  # .clamp(0, 1)
    image = F.relu(image - min_val) - F.relu(image - max_val)  # The missing clamp from above 

    clip_score = clipscore(image, prompt) 
    clip_score = clip_score.mean() 
    clip_score = str(round(clip_score.item(), 2))
    clip_values.append(clip_score)

    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8").squeeze()
    target = target.detach().cpu().permute(0, 2, 3, 1).numpy().astype("uint8").squeeze(0)

    # Calculate SSIM and PSNR
      #Skimage:
    ssim_value = ssim(image, target, win_size=3, multichannel=True)
    ssim_value = str(round(ssim_value, 5))
    ssim_values.append(ssim_value)

    psnr_value = psnr(image, target)
    psnr_value = str(round(psnr_value, 2))
    psnr_values.append(psnr_value)

      #BasicSR
    crop_border=0
    test_y_channel=False
    psnr_BasicSR = calculate_psnr(image, target, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
    ssim_BasicSR = calculate_ssim(image, target, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
    psnr_BasicSR_values.append(psnr_BasicSR)
    ssim_BasicSR_values.append(ssim_BasicSR)

    image = Image.fromarray(image)
    target = Image.fromarray(target)
    mse = mse_calc(np.array(image), np.array(target))  
    mse = str(round(mse, 2))
    mse_values.append(mse)

    # Perceptual similarity metric 
    img1 = preprocess(image).to(torch_device) 
    img2 = preprocess(target).to(torch_device) 

    distance = percep_model(img1, img2)
    distance = str(round(distance.item(), 5))
    percep_values.append(distance)

    prompt = prompt[0].replace(' ','_')
    exp_name = "_".join([filename[0], prompt, mse])

    # Check if the folder already exists
    complete_folder_path = os.path.join(args.exp_folder, args.folder_path) #Previously created test_loadalphas_SWISH_results
    if not os.path.exists(complete_folder_path):
        # Create the folder
        os.makedirs(complete_folder_path) 
    
    cvs_header = ["filename"] + ['mse'] + ['distance'] + ['clip'] + ['ssim'] + ['psnr'] + ['psnr_BasicSR'], ['ssim_BasicSR']
    create_csv_from_string(mse, distance, clip_score, ssim_value, psnr_value, psnr_BasicSR, ssim_BasicSR, cvs_header, filename[0], complete_folder_path)
    print(cond_noises)
    # Save image
    image.save(f"./{complete_folder_path}/{exp_name}.png")
  
  # NFEs and mean MSE
  # Create a mapping of values to the corresponding additions
  #value_to_add = {0: 1, 1: 1, 2: 2, 3: 2, 4: 2}
  value_to_add = {0: 1, 1: 1, 2: 2}
  NFE = np.sum([value_to_add[val] for val in policy])
  
  mean_mse = sum([float(mse) for mse in mse_values]) / len(mse_values)
  average_distance = sum([float(per) for per in percep_values]) / len(percep_values)
  clip_values = sum([float(clip) for clip in clip_values]) / len(clip_values)

  # Define the file path for the text file
  txt_filename = os.path.join(complete_folder_path, "MSE_NFEs.txt")
  #Write NFE and mean_mse to the text file
  with open(txt_filename, 'w') as txt_file:
    txt_file.write(f'NFEs: {NFE}\n')
    txt_file.write(f'mean_mse: {mean_mse}\n')
    txt_file.write(f'Policy: {policy}\n')
    txt_file.write(f'Average Distance: {average_distance}\n')
    txt_file.write(f'Average CLip Score: {clip_values}\n')
    txt_file.write(f'Average SSIM: {sum([float(s) for s in ssim_values]) / len(ssim_values)}\n')
    txt_file.write(f'Average PSNR: {sum([float(p) for p in psnr_values]) / len(psnr_values)}\n')
    txt_file.write(f'Average PSNR BasicSR: {sum([float(p) for p in psnr_BasicSR_values]) / len(psnr_BasicSR_values)}\n')
    txt_file.write(f'Average SSIM BasicSR: {sum([float(s) for s in ssim_BasicSR_values]) / len(ssim_BasicSR_values)}\n')

  return objs.avg
  
if __name__ == '__main__':
  main() 
# wandb.agent(sweep_id, function=main, count=3)