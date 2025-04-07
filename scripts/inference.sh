MASK_PATH=data/trpv1-simulated/mask.npy 
python -m instamap.reconstruction.train +dataset=reconstruct_trpv1 mask.masking_enabled=True mask.mask_path=${MASK_PATH} heterogeneity.enabled=False batch_size=1 data.limit=500 num_epochs=1 val_check_interval=0.5
python -m instamap.reconstruction.train +dataset=reconstruct_trpv1 mask.masking_enabled=False heterogeneity.enabled=True data.limit=100 val_check_interval=0.5 num_epochs=1
python -m instamap.reconstruction.train +dataset=reconstruct_trpv1 mask.masking_enabled=False heterogeneity.enabled=False data.limit=100 val_check_interval=0.5 num_epochs=1