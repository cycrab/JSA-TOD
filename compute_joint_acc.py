import json
from sklearn.metrics import f1_score, accuracy_score
import sys
import numpy as np
import csv
from dst import ignore_none, default_cleaning, IGNORE_TURNS_TYPE2, paser_bs
import argparse

def load_result(result_path):
    results=[]
    with open(result_path, 'r') as rf:
        reader=csv.reader(rf)
        for n,line in enumerate(reader):
            entry={}
            if n>0:
                if n==1:
                    field=line
                else:
                    for i,key in enumerate(field):
                        entry[key]=line[i]
                    results.append(entry)
    return results,field

def compute_jacc(data,default_cleaning_flag=True,return_db=False):
    num_turns = 0
    joint_acc = 0
    db_acc=0
    clean_tokens = ['<|endoftext|>', ]
    for turn_data in data:
        if turn_data['user']=='':
            continue
        if return_db:
            if turn_data['db']==turn_data['db_gen']:
                db_acc+=1
        turn_target = turn_data['bspn']
        turn_pred = turn_data['bspn_gen']
        turn_target = paser_bs(turn_target)
        turn_pred = paser_bs(turn_pred)
        for bs in turn_pred:
            if bs in clean_tokens + ['', ' '] or bs.split()[-1] == 'none':
                turn_pred.remove(bs)
        new_turn_pred = []
        for bs in turn_pred:
            for tok in clean_tokens:
                bs = bs.replace(tok, '').strip()
                new_turn_pred.append(bs)
        turn_pred = new_turn_pred
        turn_pred, turn_target = ignore_none(turn_pred, turn_target)
        if default_cleaning_flag:
            turn_pred, turn_target = default_cleaning(turn_pred, turn_target)
        if set(turn_target) == set(turn_pred):
            joint_acc += 1
        
        num_turns += 1

    joint_acc /= num_turns
    db_acc /= num_turns
    
    #print('joint accuracy: {}'.format(joint_acc))
    if return_db:
        return joint_acc, db_acc
    else:
        return joint_acc

def find_case(data1,data2):
    #data1: sup-only results
    #data2: semi-train results
    clean_tokens = ['<|endoftext|>', ]
    for turn_data1, turn_data2 in zip(data1,data2):
        if turn_data1['user']=='':
            continue
        assert turn_data1['bspn']==turn_data2['bspn']
        turn_target = turn_data1['bspn']
        turn_pred1 = turn_data1['bspn_gen']
        turn_pred2 = turn_data2['bspn_gen']
        turn_target = paser_bs(turn_target)
        turn_pred1 = paser_bs(turn_pred1)
        turn_pred2 = paser_bs(turn_pred2)
        for turn_pred in [turn_pred1,turn_pred2]:
            for bs in turn_pred:
                if bs in clean_tokens + ['', ' '] or bs.split()[-1] == 'none':
                    turn_pred.remove(bs)
            new_turn_pred = []
            for bs in turn_pred:
                for tok in clean_tokens:
                    bs = bs.replace(tok, '').strip()
                    new_turn_pred.append(bs)
            turn_pred = new_turn_pred
            turn_pred, turn_target = ignore_none(turn_pred, turn_target)
            turn_pred, turn_target = default_cleaning(turn_pred, turn_target)
        if set(turn_target) != set(turn_pred1) and set(turn_target)==set(turn_pred2):
            print(turn_data1['dial_id'])
            print('ground truth: \n u:{} \n b:{} \n r:{}'.format(turn_data1['user'],turn_data1['bspn'],turn_data1['resp']))
            print('sup only:\n b:{} \n r:{}'.format(turn_data1['bspn_gen'],turn_data1['resp_gen']))
            print('semi VL:\n b:{} \n r:{}'.format(turn_data2['bspn_gen'],turn_data2['resp_gen']))
            


if __name__ == "__main__":
    '''
    for s in ['10','20','30','40','50']:
        path='experiments/all_pre_{}_sd11_lr0.0001_bs2_ga16/best_score_model/result.csv'.format(s)
        results,field=load_result(path)
        joint_acc=compute_jacc(results)
        print('proportion:{}%, joint goal after pretrain:{}'.format(s,joint_acc))
    for s in ['10','20','30','40','50']:
        path='experiments/all_ST_{}_sd11_lr2e-05_bs2_ga16/best_score_model/result.csv'.format(s)
        results,field=load_result(path)
        joint_acc=compute_jacc(results)
        print('proportion:{}%, joint goal after selftrain:{}'.format(s,joint_acc))
    for s in ['10','20','30']:
        path='experiments/all_VL_{}_sd11_lr2e-05_bs2_ga16/best_score_model/result1.csv'.format(s)
        results,field=load_result(path)
        joint_acc=compute_jacc(results)
        print('proportion:{}%, joint goal after VLtrain:{}'.format(s,joint_acc))
    path='/home/liuhong/UBAR/experiments/all_semi_316_sd11_lr2e-05_bs2_ga16/best_score_model/result1.csv'
    results,field=load_result(path)
    joint_acc=compute_jacc(results)
    print('proportion:{}%, joint goal after VLtrain:{}'.format('40',joint_acc))
    '''
    path1='/home/liuhong/UBAR/experiments/all_pre_40_sd11_lr0.0001_bs2_ga16/best_score_model/result1.csv'
    path2='/home/liuhong/UBAR/experiments/all_VL_40_sd11_lr2e-05_bs2_ga16/best_score_model/result1.csv'
    results1,_=load_result(path1)
    results2,_=load_result(path2)
    find_case(results1,results2)