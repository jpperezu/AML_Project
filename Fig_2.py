import os
import glob
import torch 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

chosen_sweeps = ['ddpm_nas/test_loadalphas_CC3M_9000_v2_cost20-relu_gumbel_entropy_temp_20steps/im-s_512_CC3M_9000_v2_cost-relu_gumbel_softmax_perceptual_entropy_temp_lmpercep_lr_0.05_optim_Adam_numInfSteps_20_gamma_0.001_temp_1_l-cost_0.1_l-latent_1_l-entropy_-0.1_tau_1_wandering-sweep-350', 
'ddpm_nas/test_loadalphas_CC3M_9000_v2_cost20-relu_gumbel_entropy_temp_20steps/im-s_512_CC3M_9000_v2_cost-relu_gumbel_softmax_perceptual_entropy_temp_lr_0.1_optim_SGD_numInfSteps_20_gamma_0.001_temp_1_l-cost_0.1_l-latent_1_l-entropy_0.1_tau_0.5_giddy-sweep-44',
'ddpm_nas/test_loadalphas_CC3M_9000_v2_cost20-relu_gumbel_entropy_temp_20steps/im-s_512_CC3M_9000_v2_cost-relu_gumbel_softmax_perceptual_entropy_temp_lr_0.005_optim_SGD_numInfSteps_20_gamma_0.001_temp_1_l-cost_10_l-latent_10_l-entropy_10_tau_0.5_sweet-sweep-4',
'ddpm_nas/test_loadalphas_CC3M_9000_v2_cost20-relu_gumbel_entropy_temp_20steps/im-s_512_CC3M_9000_v2_cost-relu_gumbel_softmax_perceptual_entropy_temp_lr_0.05_optim_SGD_numInfSteps_20_gamma_0.001_temp_1_l-cost_10_l-latent_10_l-entropy_-10_tau_0.8_vague-sweep-5',
'ddpm_nas/test_loadalphas_CC3M_9000_v2_cost20-relu_gumbel_entropy_temp_20steps/im-s_512_CC3M_9000_v2_cost-relu_gumbel_softmax_perceptual_entropy_temp_lmpercep_lr_0.001_optim_Adam_numInfSteps_20_gamma_0.001_temp_1_l-cost_0.1_l-latent_10_l-entropy_-1_tau_0.8_wise-sweep-311',
'ddpm_nas/test_loadalphas_CC3M_9000_v2_cost20-relu_gumbel_entropy_temp_20steps/im-s_512_CC3M_9000_v2_cost-relu_gumbel_softmax_perceptual_entropy_temp_lr_0.01_optim_SGD_numInfSteps_20_gamma_0.001_temp_1_l-cost_0.01_l-latent_1_l-entropy_-10_tau_1_sunny-sweep-38',
'ddpm_nas/test_loadalphas_CC3M_9000_v2_cost20-relu_gumbel_entropy_temp_20steps/im-s_512_CC3M_9000_v2_cost-relu_gumbel_softmax_perceptual_entropy_temp_lmpercep_lr_0.1_optim_Lion_numInfSteps_20_gamma_0.001_temp_1_l-cost_0.01_l-latent_0.1_l-entropy_0.1_tau_0.1_genial-sweep-318'
]

cond, unc, cfg = [], [], []

for k, file in enumerate(chosen_sweeps):
    alpha_file = os.path.join(file,'alphas_e*.pt')
    alpha_file = glob.glob(alpha_file)[0]
    rawalphas =  torch.load(alpha_file)
    softalphas = torch.nn.functional.softmax(rawalphas, dim=-1)
    num_steps = softalphas.shape[0]
    num_columns = softalphas.shape[1]

    for i in range(num_columns):
        if i == 0:
            unc.append(softalphas[:, i].view(num_steps,1))
        elif i == 1:
            cond.append(softalphas[:, i].view(num_steps,1))
        else:
            cfg.append(softalphas[:, i].view(num_steps,1))
        
labels = ['u', 'c', 'cfg']
for i in range(num_columns):

    if i == 0:
        mean_unc = torch.concat(unc, dim=-1).mean(dim=-1)
        # plt.plot(mean_unc, label=labels[i])
    elif i == 1:
        mean_cond = torch.concat(cond, dim=-1).mean(dim=-1)
        # plt.plot(mean_cond, label=labels[i])
    else:
        mean_cfg = torch.concat(cfg, dim=-1).mean(dim=-1)
        # plt.plot(mean_cfg, label=labels[i])


# Create a DataFrame for Seaborn
data = {'u': mean_unc.numpy(),
        '------': mean_cond.numpy(),
        '--------': mean_cfg.numpy()}

# Set seaborn style
sns.set(style="whitegrid")

plt.figure(figsize=(10, 8))
df = pd.DataFrame(data)
# Convert lists to tensors
unc = torch.cat(unc, dim=-1)
cond = torch.cat(cond, dim=-1)
cfg = torch.cat(cfg, dim=-1)

# Calculate mean and standard deviation
mean_unc = unc.mean(dim=-1).numpy()
std_unc = unc.std(dim=-1).numpy()

mean_cond = cond.mean(dim=-1).numpy()
std_cond = cond.std(dim=-1).numpy()

mean_cfg = cfg.mean(dim=-1).numpy()
std_cfg = cfg.std(dim=-1).numpy()

# Create a Seaborn line plot with shaded standard deviation
# sns.set_theme()
palette = sns.color_palette("husl", n_colors=3)
plt.figure(figsize=(10, 7))

sns.lineplot(data=mean_unc, label='Unconditional', linewidth=3)
plt.fill_between(range(len(mean_unc)), mean_unc - std_unc, mean_unc + std_unc, alpha=0.2)

sns.lineplot(data=mean_cond, label='Conditional', linewidth=3)
plt.fill_between(range(len(mean_cond)), mean_cond - std_cond, mean_cond + std_cond, alpha=0.2)

sns.lineplot(data=mean_cfg, label='CFG', linewidth=3)
plt.fill_between(range(len(mean_cfg)), mean_cfg - std_cfg, mean_cfg + std_cfg, alpha=0.2)

# plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)
plt.xlim(1, len(df))
plt.legend(fontsize=22)
plt.tick_params(axis='x', labelsize=13)
plt.tick_params(axis='y', labelsize=13)
plt.xlabel('Step Index', fontsize=15)
plt.ylabel('Score', fontsize=15)
# Create a Seaborn line plot

plt.savefig('plot_alphas.png')
plt.show()
