python -m mask_cyclegan_vc.test \
    --name uas \
    --save_dir results/ \
    --preprocessed_data_dir uas_preprocessed_test/\
    --gpu_ids 0 \
    --speaker_A_id M09 \
    --speaker_B_id CM10 \
    --ckpt_dir results/uas/ckpts \
    --load_epoch 800 \
    --model_name generator_A2B \
