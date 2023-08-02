DATASET=ScanNet_00_reg

# Training a Vanilla NeRF as teacher model
python run_nerf.py cfgs/paper/pretrain/$DATASET.yaml

# Extracting occupancy grid from teacher model
python build_occupancy_tree.py cfgs/paper/pretrain_occupancy/$DATASET.yaml

# Distilling tacher into KiloNeRF model
python local_distill.py cfgs/paper/distill/$DATASET.yaml

# Fine-tuning KiloNeRF model
python run_nerf.py cfgs/paper/finetune/$DATASET.yaml


DATASET2=ScanNet_00_seg

# Training a Vanilla NeRF as teacher model
python run_nerf.py cfgs/paper/pretrain/$DATASET2.yaml

# Extracting occupancy grid from teacher model
python build_occupancy_tree.py cfgs/paper/pretrain_occupancy/$DATASET2.yaml

# Distilling tacher into KiloNeRF model
python local_distill.py cfgs/paper/distill/$DATASET2.yaml

# Fine-tuning KiloNeRF model
python run_nerf.py cfgs/paper/finetune/$DATASET2.yaml
