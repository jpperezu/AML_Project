#Choose the policy you want to use (make sure to adapt num_inference_steps to your policy)  Here we use #AG 0.9994 (32 NFEs)

CUDA_VISIBLE_DEVICES=0 python main.py --num_inference_steps 20 --base_folder demo_image --policy 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 --filename 0 --prompt 'Lovely white fondant-covered cake with a purple ribbon'
