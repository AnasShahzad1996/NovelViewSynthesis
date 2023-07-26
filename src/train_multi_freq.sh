DATASET=TanksAndTemple_Barn_MultiRes
set -euxo pipefail
# Training a Vanilla NeRF as teacher model
#time python run_nerf.py cfgs/paper/pretrain/$DATASET.yaml

# Extracting occupancy grid from teacher model
#time python build_occupancy_tree.py cfgs/paper/pretrain_occupancy/$DATASET.yaml

# Scale down occupancy grid for testing and visualization
#time python multiscale_occupancy.py $DATASET

# Distilling teacher into KiloNeRF model
#time python local_distill.py cfgs/paper/distill/${DATASET}.yaml
#time python local_distill.py cfgs/paper/distill/${DATASET}_LowFreq.yaml
#time python local_distill.py cfgs/paper/distill/${DATASET}_LowFreq_Small.yaml

# Fine-tuning KiloNeRF model
time python run_nerf.py cfgs/paper/finetune/${DATASET}.yaml
time python run_nerf.py cfgs/paper/finetune/${DATASET}_LowFreq.yaml
time python run_nerf.py cfgs/paper/finetune/${DATASET}_LowFreq_Small.yaml
