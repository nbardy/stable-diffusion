python3 main.py \
    -t \
    --base configs/stable-diffusion/pokemon.yaml \
    --gpus 0 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 10 \
    --finetune_from /home/paperspace/.cache/huggingface/hub/models--CompVis--stable-diffusion-v-1-4-original/snapshots/0834a76f88354683d3f7ef271cadd28f4757a8cc/sd-v1-4-full-ema.ckpt
