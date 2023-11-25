#Fig 1:
CUDA_VISIBLE_DEVICES=0 python Fig_1_4.py --num_inference_steps 20 --base_folder D_20_CFG --policy 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 --json Fig_1.json
CUDA_VISIBLE_DEVICES=1 python Fig_1_4.py --num_inference_steps 15 --base_folder D_15_CFG --policy 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 --json Fig_1.json
CUDA_VISIBLE_DEVICES=2 python Fig_1_4.py --num_inference_steps 20 --base_folder D_AG_10_CFG_10_C --policy 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 --json Fig_1.json

#Fig 4:

# 20 CFG (40 NFEs)
CUDA_VISIBLE_DEVICES=4 python Fig_1_4.py --num_inference_steps 20 --base_folder D_20_CFG --json Fig_4.json
# 16 CFG (32 NFEs)
CUDA_VISIBLE_DEVICES=5 python Fig_1_4.py --num_inference_steps 16 --base_folder D_16_CFG --policy 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 --json Fig_4.json
# 15 CFG (30 NFEs)
CUDA_VISIBLE_DEVICES=6 python Fig_1_4.py --num_inference_steps 15 --base_folder D_15_CFG --policy 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 --json Fig_4.json
#AG 0.9994 (32 NFEs)
CUDA_VISIBLE_DEVICES=7 python Fig_1_4.py --num_inference_steps 20  --base_folder D_AG_0.9994_CFG --policy 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 --json Fig_4.json
#AG 0.9990 (30 NFEs)
CUDA_VISIBLE_DEVICES=0 python Fig_1_4.py --num_inference_steps 20 --base_folder D_AG_0.9990_CFG --policy 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 --json Fig_4.json

