import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set seaborn style
sns.set(style="whitegrid")

# Custom sorting key function to extract the final number
def get_final_number(folder_path):
    return int(folder_path.split("_")[-2])

def get_final_number2(folder_path):
    return int(folder_path.split("_")[-1])

metric={'MSE':1, 'DISTANCE':3, 'PSNR':-2, 'SSIM':-1}

#******************************************************************* SSIM **********************************************************************************************************

# --------------------------------------Define the parent folder path NAS----------------------------------------------------------
parent_folder_exp = 'ddpm_nas/RESULTS_ANGY_CC3M_9000_v2_cost30-relu_gumbel_softmax_perceptual_entropy_temp_20steps'

child_folders = [f.path for f in os.scandir(parent_folder_exp) if f.is_dir()]
                  
x, y, labels = [], [], []

# Loop through each child folder
for k, folder in enumerate(child_folders):
    txt_file = os.path.join(folder, 'MSE_NFEs.txt')

    if os.path.isfile(txt_file):
        with open(txt_file, 'r') as txt:
            lines = txt.readlines()

        # Extract "NFEs" and "mean_mse" values
        nfe_line = lines[0].strip()
        mean_mse_line = lines[metric['SSIM']].strip() 

        # Split the lines to get the values
        nfe_value = int(nfe_line.split(": ")[1])
        mean_mse_value = float(mean_mse_line.split(": ")[-1])
        # Round mean_mse to 2 decimal places
        mean_mse_value_rounded = round(mean_mse_value, 2)
        # Sample data
        x.append(nfe_value)
        y.append(mean_mse_value_rounded)

        subfolder = folder.split('/')[-1]
        current_sweep = subfolder[subfolder.find('sweep-') + 6:]
        # Specify labels for each point
        labels.append('')

fig, ax = plt.subplots(figsize=(15, 8))
sns.scatterplot(x=x, y=y, s=70, label=r'Policies $\epsilon_{cfg}(x_{t}, c, 7.5)$', color='steelblue')

# Add labels for each point
for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=20)

# breakpoint()

# x_extra=[36, 38]
# y_extra=[0.75, 0.76]
# labels=['', '']

x_extra =[27, 30, 29, 31, 30, 31, 28, 28, 31, 31, 31, 29, 28, 32, 32, 34, 28, 29, 28] #Manual, porque cargado no me estaba sirviendo 
y_extra=[0.5, 0.56, 0.52, 0.54, 0.52, 0.6, 0.5, 0.48, 0.66, 0.51, 0.5, 0.5, 0.49, 0.51, 0.56, 0.56, 0.5, 0.67, 0.5]
labels= ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

# breakpoint()

sns.scatterplot(x=x_extra, y=y_extra, s=70, label=r'Policies $\epsilon_{cfg}(x_{t}, c, \neq 7.5)$', color='plum')

for i, label in enumerate(labels):
    plt.annotate('', xy=(x_extra[i], y_extra[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=20)


# --------------------------------------Define the parent folder path FULL CFG BASELINE ----------------------------------------------------------
parent_folder = 'ddpm_nas/alphas_threshold'
child_folders = [f.path for f in os.scandir(parent_folder) if f.is_dir() and f.name.startswith("test_DPM_")]
child_folders = sorted(child_folders, key=get_final_number2)

line_x, line_y = [], []

# Loop through each child folder
for k, folder in enumerate(child_folders):
    txt_file = os.path.join(folder, 'MSE_NFEs.txt')

    if os.path.isfile(txt_file):
        with open(txt_file, 'r') as txt:
            lines = txt.readlines()
        
        # Extract "NFEs" and "mean_mse" values
        nfe_line = lines[0].strip()
        distance_line = lines[metric['SSIM']].strip() 

        # Split the lines to get the values
        nfe_value = int(nfe_line.split(": ")[1])
        distance_value = float(distance_line.split(": ")[1])
        # Round distance to 2 decimal places
        distance_value_rounded = round(distance_value, 4)
        # Sample data
        line_x.append(nfe_value)
        line_y.append(distance_value_rounded)

ax.plot(line_x, line_y, label='CFG', color='salmon', linestyle='solid', linewidth=3)


# --------------------------------------Define the parent folder path TRESHOLD 0.9996 ----------------------------------------------------------

parent_folder = 'ddpm_nas/alphas_threshold'
child_folders_c0_2 = [f.path for f in os.scandir(parent_folder) if f.is_dir() and f.name.startswith("test_c0.3")]
child_folders_c0_2 = sorted(child_folders_c0_2, key=get_final_number)

line_x, line_y = [], []

# Loop through each child folder
for k, folder in enumerate(child_folders_c0_2):
    txt_file = os.path.join(folder, 'MSE_NFEs.txt')

    if os.path.isfile(txt_file):
        with open(txt_file, 'r') as txt:
            lines = txt.readlines()

        # Extract "NFEs" and "mean_mse" values
        nfe_line = lines[0].strip()
        mean_mse_line = lines[metric['SSIM']].strip() 

        # Split the lines to get the values
        nfe_value = int(nfe_line.split(": ")[1])
        mean_mse_value = float(mean_mse_line.split(": ")[-1])
        # Round mse to 2 decimal places
        mean_mse_value_rounded = round(mean_mse_value, 4)
        # Sample data
        line_x.append(nfe_value)
        line_y.append(mean_mse_value_rounded)

# Line plot
sns.lineplot(x=line_x, y=line_y, label=r'AG $\bar{\gamma}=0.9996$', color='orange', linestyle='dotted', ax=ax, linewidth=5) #c0.2


# --------------------------------------Define the parent folder path TRESHOLD 0.9994 ----------------------------------------------------------

parent_folder = 'ddpm_nas/alphas_threshold'

child_folders_c0_3 = [f.path for f in os.scandir(parent_folder) if f.is_dir() and f.name.startswith("test_c0.4")]
child_folders_c0_3 = sorted(child_folders_c0_3, key=get_final_number)

line_x, line_y = [], []

# Loop through each child folder
for k, folder in enumerate(child_folders_c0_3):
    txt_file = os.path.join(folder, 'MSE_NFEs.txt')

    if os.path.isfile(txt_file):
        with open(txt_file, 'r') as txt:
            lines = txt.readlines()

        # Extract "NFEs" and "mean_mse" values
        nfe_line = lines[0].strip()
        mean_mse_line = lines[metric['SSIM']].strip() 

        # Split the lines to get the values
        nfe_value = int(nfe_line.split(": ")[1])
        mean_mse_value = float(mean_mse_line.split(": ")[-1])
        # Round mse to 2 decimal places
        mean_mse_value_rounded = round(mean_mse_value, 4)
        # Sample data
        line_x.append(nfe_value)
        line_y.append(mean_mse_value_rounded)

# Line plot
sns.lineplot(x=line_x, y=line_y, label=r'AG $\bar{\gamma}=0.9994$', color='g', linestyle='dotted', ax=ax, linewidth=5) #c0.3


# --------------------------------------Define the parent folder path TRESHOLD 0.9990 ----------------------------------------------------------

parent_folder = 'ddpm_nas/alphas_threshold'
child_folders_c0_4 = [f.path for f in os.scandir(parent_folder) if f.is_dir() and f.name.startswith("test_c0.5")]
child_folders_c0_4 = sorted(child_folders_c0_4, key=get_final_number)

line_x, line_y = [], []

# Loop through each child folder
for k, folder in enumerate(child_folders_c0_4):
    txt_file = os.path.join(folder, 'MSE_NFEs.txt')

    if os.path.isfile(txt_file):
        with open(txt_file, 'r') as txt:
            lines = txt.readlines()

        # Extract "NFEs" and "mean_mse" values
        nfe_line = lines[0].strip()
        mean_mse_line = lines[metric['SSIM']].strip() 

        # Split the lines to get the values
        nfe_value = int(nfe_line.split(": ")[1])
        mean_mse_value = float(mean_mse_line.split(": ")[-1])
        # Round mse to 2 decimal places
        mean_mse_value_rounded = round(mean_mse_value, 4)
        # Sample data
        line_x.append(nfe_value)
        line_y.append(mean_mse_value_rounded)

# Line plot
sns.lineplot(x=line_x, y=line_y, label=r'AG $\bar{\gamma}=0.9990$', color='paleturquoise', linestyle='dotted', ax=ax, linewidth=5) #c0.4

#ax.set_title('TG Performance at Different NFEs Values')
plt.legend(fontsize=20, loc='lower right')
ax.set_xlabel('NFEs',fontsize=17) #30
plt.tick_params(axis='x', labelsize=15)
ax.set_ylabel('SSIM',fontsize=17)
plt.tick_params(axis='y', labelsize=15)
plt.savefig('Fig_5.png')