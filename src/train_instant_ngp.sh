DATASET=Synthetic_NeRF_Lego_Norm
# Define dataset_path and run scaling operation required for instant ngp

python scale_for_ingp.py data/nsvf/Synthetic_NeRF/Lego

# Training Instant Ngp as teacher model
# python train.py --root_dir $dataset_path --exp_name Lego
python ngp_pl/train.py --root_dir data/nsvf/Synthetic_NeRF/Lego_norm --exp_name Lego

# Trained using ngp_pl github repository (https://github.com/kwea123/ngp_pl) and loading the saved checkpoint
cp ngp_pl/ckpts/nsvf/Lego/epoch\=29_slim.ckpt logs/paper/pretrain/Synthetic_NeRF_Lego_Norm/

# Extracting occupancy grid from teacher model
python build_occupancy_tree.py cfgs/paper/pretrain_occupancy/$DATASET.yaml

# Distilling teacher into KiloNeRF model
python local_distill.py cfgs/paper/distill/$DATASET.yaml

# Fine-tuning KiloNeRF model
python run_nerf.py cfgs/paper/finetune/$DATASET.yaml
