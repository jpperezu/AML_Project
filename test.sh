
#For a particular sweep resut:

CUDA_VISIBLE_DEVICES=0 python test_AML.py --test_data test_1000_dataset --test_json 'test_1000.json' --exp_folder TEST_RESULTS --folder_path im-s_512_CC3M_9000_v2_cost-relu_gumbel_softmax_perceptual_entropy_temp_lmpercep_lr_0.05_optim_Adam_numInfSteps_20_gamma_0.001_temp_1_l-cost_0.1_l-latent_1_l-entropy_-0.1_tau_1_wandering-sweep-350 --load_folder test_loadalphas_CC3M_9000_v2_cost20-relu_gumbel_entropy_temp_20steps/im-s_512_CC3M_9000_v2_cost-relu_gumbel_softmax_perceptual_entropy_temp_lmpercep_lr_0.05_optim_Adam_numInfSteps_20_gamma_0.001_temp_1_l-cost_0.1_l-latent_1_l-entropy_-0.1_tau_1_wandering-sweep-350

CUDA_VISIBLE_DEVICES=0 python test_AML.py --test_data ddpm_nas/test_1000_dataset --test_json 'ddpm_nas/test_1000.json' --exp_folder TEST_RESULTS --folder_path im-s_512_CC3M_9000_v2_cost-relu_gumbel_softmax_perceptual_entropy_temp_lmpercep_lr_0.05_optim_Adam_numInfSteps_20_gamma_0.001_temp_1_l-cost_0.1_l-latent_1_l-entropy_-0.1_tau_1_wandering-sweep-350 --load_folder ddpm_nas/test_loadalphas_CC3M_9000_v2_cost20-relu_gumbel_entropy_temp_20steps/im-s_512_CC3M_9000_v2_cost-relu_gumbel_softmax_perceptual_entropy_temp_lmpercep_lr_0.05_optim_Adam_numInfSteps_20_gamma_0.001_temp_1_l-cost_0.1_l-latent_1_l-entropy_-0.1_tau_1_wandering-sweep-350
