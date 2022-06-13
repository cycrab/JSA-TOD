ratio=$2
path1=experiments_21/all_turn_level_pre_${ratio}_act_sd11_lr0.0001_bs8_ga4/best_score_model
path2=experiments_21/all_turn_level_pre_pos${ratio}_act_sd11_lr0.0001_bs8_ga4/best_loss_model

python main_turn_level.py -mode semi_jsa\
    -cfg PrioriModel_path=$path1 PosteriorModel_path=$path2 \
    lr=1e-4 \
    turn_level=True\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=40\
    cuda_device=$1\
    spv_proportion=$ratio \
    model_act=True \