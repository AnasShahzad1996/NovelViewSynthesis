DATASET=Synthetic_NeRF_Lego_Norm
# Define dataset_path and run scaling operation required for instant ngp

# python scale_for_ingp.py $dataset_path

# Training Instant Ngp as teacher model
# Trained using ngp_pl github repository and loading the saved checkpoint

# Extracting occupancy grid from teacher model
python build_occupancy_tree.py cfgs/paper/pretrain_occupancy/$DATASET.yaml

# Distilling tacher into KiloNeRF model
python local_distill.py cfgs/paper/distill/$DATASET.yaml

# Fine-tuning KiloNeRF model
python run_nerf.py cfgs/paper/finetune/$DATASET.yaml
