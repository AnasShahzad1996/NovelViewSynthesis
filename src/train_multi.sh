DATASET=Synthetic_NeRF_Lego_MultiRes_Low

# Training a Vanilla NeRF as teacher model
#time python run_nerf.py cfgs/paper/pretrain/$DATASET.yaml

# Extracting occupancy grid from teacher model
#time python build_occupancy_tree.py cfgs/paper/pretrain_occupancy/$DATASET.yaml

# Distilling tacher into KiloNeRF model
time python multiscale_local_distill.py cfgs/paper/distill/$DATASET.yaml

# Fine-tuning KiloNeRF model
time python multiscale_run_nerf.py cfgs/paper/finetune/$DATASET.yaml
