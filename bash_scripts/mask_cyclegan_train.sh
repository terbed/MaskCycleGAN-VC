# Sample training script to convert between VCC2SF3 and VCC2TF1
# Continues training from epoch 500

python -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_SF1_TF1 \
    --seed 0 \
    --save_dir results/ \
    --preprocessed_data_dir preprocessed/ \
    --speaker_A_id SF1 \
    --speaker_B_id TF1 \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --epochs_per_plot 10000 \
    --decay_after 2e5 \
    --stop_identity_after 1e4 \
    --batch_size 1 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 32 \
    --gpu_ids 0 \
    --cycle_loss_lambda 1. \
    --start_epoch 100

