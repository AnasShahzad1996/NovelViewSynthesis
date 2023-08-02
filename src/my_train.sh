DATASET=Synthetic_NeRF_Lego

# Distilling tacher into KiloNeRF model
python local_distill.py cfgs/paper/distill/$DATASET.yaml