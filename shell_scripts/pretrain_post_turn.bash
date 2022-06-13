python main_turn_level.py -mode pretrain\
    -cfg  lr=1e-4\
    turn_level=True\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=50\
    cuda_device=$1\
    spv_proportion=$2\
    model_act=True \
    posterior_train=True\
    save_type=min_loss