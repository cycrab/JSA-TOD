from cProfile import label
from config import global_config as cfg
#from eval import MultiWozEvaluator
from reader import MultiWozReader
from transformers import GPT2Tokenizer
import json, random
import ontology
import matplotlib.pyplot as plt

#tokenizer=GPT2Tokenizer.from_pretrained('experiments_21/turn-level-DS/best_score_model')
#reader = MultiWozReader(tokenizer)
#evaluator = MultiWozEvaluator(reader)
def variance_compare():#INFO:root:  Num Batches = 279
    path_jsa = 'experiments_21/jsa_var_new.json'
    path_vl = 'experiments_21/vl_var.json'
    #result_path =  'experiments_21/result/vl_11/result5_.json'
    jsa_data=json.load(open(path_jsa, 'r', encoding='utf-8'))
    vl_data=json.load(open(path_vl, 'r', encoding='utf-8'))
    jsa_data_new=[]
    vl_data_new=[]
    agg_count = 5
    jsa_count = 0 
    jsa_sum = 0
    vl_count = 0
    vl_sum = 0
    for i in range(len(vl_data)):
        if jsa_data[i]!=0:
            jsa_count = jsa_count + 1
            jsa_sum = jsa_sum + jsa_data[i]
            if jsa_count==agg_count:
                jsa_data_new.append(jsa_sum)
                jsa_count = 0
                jsa_sum = 0
        if vl_data[i]!=0:
            vl_count = vl_count + 1
            vl_sum = vl_sum + vl_data[i]
            if vl_count==agg_count:
                vl_data_new.append(vl_sum)
                vl_count = 0
                vl_sum = 0
    max_jsa=0
    jsa_sum=0
    max_vl=0
    vl_sum=0
    upp_bound = int(500/agg_count)
    lower_bound = int(1250/agg_count)
    for i in range(upp_bound,lower_bound):
        jsa_sum = jsa_sum + jsa_data_new[i]
        vl_sum = vl_sum + vl_data_new[i]
        #if jsa_data[i]>max_jsa:
        #    max_jsa=jsa_data[i]
        #if vl_data[i]>max_vl:
        #    max_vl=vl_data[i]
    for i in range(upp_bound,lower_bound):
        jsa_data_new[i] = jsa_data_new[i]/jsa_sum
        vl_data_new[i] = vl_data_new[i]/vl_sum
    #plt.subplot(2,1,1)
    plt.plot(jsa_data_new[upp_bound:lower_bound],label='JSA',color='blue')
    #plt.subplot(2,1,2)
    plt.plot(vl_data_new[upp_bound:lower_bound],linestyle='--',label='Variational',color='r')
    plt.xlabel('Training iterations',fontsize =22)
    plt.ylabel('Gradient norm of inference model',fontsize =22)
    plt.legend(fontsize =22)
    plt.xticks(fontsize =20)
    plt.yticks(fontsize =20)
    plt.show()
    print(1) 

def compare_offline_result(path1, path2, show_num=10):
    succ1_unsuc2=[]
    succ2_unsuc1=[]
    data1=json.load(open(path1, 'r', encoding='utf-8'))
    data2=json.load(open(path2, 'r', encoding='utf-8'))
    dials1=evaluator.pack_dial(data1)
    dials2=evaluator.pack_dial(data2)
    counts = {}
    for req in evaluator.requestables:
        counts[req+'_total'] = 0
        counts[req+'_offer'] = 0
    dial_id_list=random.sample(reader.test_list, show_num)
    dial_samples=[]
    for dial_id in dials1:
        dial1=dials1[dial_id]
        dial2=dials2[dial_id]
        if dial_id+'.json' in dial_id_list:
            dial_samples.append({'dial1':dial1, 'dial2':dial2})
        reqs = {}
        goal = {}
        if '.json' not in dial_id and '.json' in list(evaluator.all_data.keys())[0]:
            dial_id = dial_id + '.json'
        for domain in ontology.all_domains:
            if evaluator.all_data[dial_id]['goal'].get(domain):
                true_goal = evaluator.all_data[dial_id]['goal']
                goal = evaluator._parseGoal(goal, true_goal, domain)
        # print(goal)
        for domain in goal.keys():
            reqs[domain] = goal[domain]['requestable']

        # print('\n',dial_id)
        success1, match1, _, _ = evaluator._evaluateGeneratedDialogue(dial1, goal, reqs, counts)
        success2, match2, _, _ = evaluator._evaluateGeneratedDialogue(dial2, goal, reqs, counts)
        if success1 and not success2:
            succ1_unsuc2.append(dial_id)
        elif success2 and not success1:
            succ2_unsuc1.append(dial_id)
    print('Success in data1 and unsuccess in data2:', len(succ1_unsuc2))#, succ1_unsuc2)
    print('Success in data2 and unsuccess in data1:', len(succ2_unsuc1))#, succ2_unsuc1)
    examples=[]
    for item in dial_samples:
        dialog=[]
        for turn1, turn2 in zip(item['dial1'], item['dial2']):
            if turn1['user']=='':
                continue
            entry={'user': turn1['user'], 'Oracle':turn1['resp'], 'Sup':turn1['resp_gen'], 'RL':turn2['resp_gen']}
            dialog.append(entry)
        examples.append(dialog)
    json.dump(examples, open('analysis/examples.json', 'w'), indent=2)
            


def compare_online_result(path1, path2):
    succ1_unsuc2=[]
    succ2_unsuc1=[]
    data1=json.load(open(path1, 'r', encoding='utf-8'))
    data2=json.load(open(path2, 'r', encoding='utf-8'))
    counts = {}
    for req in evaluator.requestables:
        counts[req+'_total'] = 0
        counts[req+'_offer'] = 0
    flag1=0
    flag2=0
    for i, dial_id in enumerate(reader.test_list):
        reqs = {}
        goal = {}
        dial1=data1[i]
        dial2=data2[i]
        if isinstance(dial1, list):
            data1[i]={dial_id:dial1}
            flag1=1
        elif isinstance(dial1, dict):
            dial1=dial1[dial_id]
        
        if isinstance(dial2, list):
            data2[i]={dial_id:dial2}
            flag2=1
        elif isinstance(dial2, dict):
            dial2=dial2[dial_id]

        init_goal=reader.data[dial_id]['goal']
        for domain in ontology.all_domains:
            if init_goal.get(domain):
                true_goal = init_goal
                goal = evaluator._parseGoal(goal, true_goal, domain)
        for domain in goal.keys():
            reqs[domain] = goal[domain]['requestable']
        success1, match2, _, _ = evaluator._evaluateGeneratedDialogue(dial1, goal, reqs, counts)
        success2, match2, _, _ = evaluator._evaluateGeneratedDialogue(dial2, goal, reqs, counts)
        if success1 and not success2:
            succ1_unsuc2.append(dial_id)
        elif success2 and not success1:
            succ2_unsuc1.append(dial_id)
    print('Success in data1 and unsuccess in data2:', len(succ1_unsuc2), succ1_unsuc2)
    print('Success in data2 and unsuccess in data1:', len(succ2_unsuc1), succ2_unsuc1)
    if flag1:
        json.dump(data1, open(path1, 'w'), indent=2)
    if flag2:
        json.dump(data2, open(path2, 'w'), indent=2)

def group_act(act):
    for domain in act:
        for intent, sv in act[domain].items():
            act[domain][intent]=set(sv)
    return act

def find_unseen_usr_act(path1=None, path2=None):
    data=json.load(open('data/multi-woz-2.1-processed/data_for_us.json', 'r', encoding='utf-8'))
    train_act_pool=[]
    unseen_act_pool=[]
    unseen_dials=[]
    for dial_id, dial in data.items():
        if dial_id in reader.train_list:
            for turn in dial:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    train_act_pool.append(user_act)
    for dial_id, dial in data.items():
        if dial_id in reader.test_list:# or dial_id in reader.dev_list:
            unseen_turns=0
            for turn in dial:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    unseen_act_pool.append(user_act)
                    unseen_turns+=1
            if unseen_turns>0:
                unseen_dials.append(dial_id)
    print('Total training acts:', len(train_act_pool), 'Unseen acts:', len(unseen_act_pool))
    print('Unseen dials:',len(unseen_dials))
    if path1 and path2:
        data1=json.load(open(path1, 'r', encoding='utf-8'))
        data2=json.load(open(path2, 'r', encoding='utf-8'))
        unseen_act_pool1=[]
        unseen_act_pool2=[]
        for dial1, dial2 in zip(data1, data2):
            dial1=list(dial1.values())[0]
            dial2=list(dial2.values())[0]
            for turn in dial1:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    unseen_act_pool1.append(user_act)
            for turn in dial2:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    unseen_act_pool2.append(user_act)
        print('Unseen acts in path1:', len(unseen_act_pool1))
        print('Unseen acts in path2:', len(unseen_act_pool2))
    return unseen_dials

def find_unseen_sys_act():
    data=json.load(open('data/multi-woz-2.1-processed/data_for_us.json', 'r', encoding='utf-8'))
    train_act_pool=[]
    unseen_act_pool=[]
    unseen_dials={}
    for dial_id, dial in data.items():
        if dial_id in reader.train_list:
            for turn in dial:
                sys_act=reader.aspan_to_act_dict(turn['sys_act'], 'sys')
                sys_act=group_act(sys_act)
                if sys_act not in train_act_pool:
                    train_act_pool.append(sys_act)
    for dial_id, dial in data.items():
        if dial_id in reader.test_list:# or dial_id in reader.dev_list:
            unseen_turns=[]
            for turn_id, turn in enumerate(dial):
                sys_act=reader.aspan_to_act_dict(turn['sys_act'], 'sys')
                sys_act=group_act(sys_act)
                if sys_act not in train_act_pool:
                    unseen_act_pool.append(sys_act)
                    unseen_turns.append(turn_id)
            if len(unseen_turns)>0:
                unseen_dials[dial_id]=unseen_turns
    print('Total training acts:', len(train_act_pool), 'Unseen acts:', len(unseen_act_pool))
    print('Unseen dials:',len(unseen_dials))
    json.dump(unseen_dials, open('analysis/unseen_turns.json', 'w'), indent=2)

    return unseen_dials

def calculate_unseen_acc(unseen_turns, path1=None, path2=None):
    data1=json.load(open(path1, 'r', encoding='utf-8'))
    data2=json.load(open(path2, 'r', encoding='utf-8'))
    total_unseen_act=0
    sup_acc=0
    rl_acc=0
    tp1=0
    fp1=0
    tp2=0
    fp2=0
    count=0
    for dial_id in unseen_turns:
        for t in unseen_turns[dial_id]:
            count+=1
    print('Total unseen act:', count)
    for turn1, turn2 in zip(data1, data2):
        dial_id=turn1['dial_id']+'.json'
        if dial_id in unseen_turns and turn1['user']!='' and turn1['turn_num'] in unseen_turns[dial_id]:
            total_unseen_act+=1
            #unseen_turns[dial_id]=unseen_turns[dial_id][1:]
            oracle_act=group_act(reader.aspan_to_act_dict(turn1['aspn'], side='sys'))
            sup_act=group_act(reader.aspan_to_act_dict(turn1['aspn_gen'], side='sys'))
            rl_act=group_act(reader.aspan_to_act_dict(turn2['aspn_gen'], side='sys'))
            if sup_act==oracle_act:
                sup_acc+=1
            if rl_act==oracle_act:
                rl_acc+=1
            for domain in sup_act:
                for intent, slots in sup_act[domain].items():
                    if domain not in oracle_act or intent not in oracle_act[domain]:
                        fp1+=len(slots)
                        continue
                    for slot in slots:
                        if slot in oracle_act[domain][intent]:
                            tp1+=1
                        else:
                            fp1+=1
            for domain in rl_act:
                for intent, slots in rl_act[domain].items():
                    if domain not in oracle_act or intent not in oracle_act[domain]:
                        fp2+=len(slots)
                        continue
                    for slot in slots:
                        if slot in oracle_act[domain][intent]:
                            tp2+=1
                        else:
                            fp2+=1
    print('Total unseen acts:{}, Sup acc:{}, RL acc:{}'.format(total_unseen_act, sup_acc, rl_acc))
    print(tp1, fp1, tp1/(tp1+fp1))
    print(tp2, fp2, tp2/(tp2+fp2))

def extract_goal():
    data=json.load(open('data/multi-woz-2.1-processed/data_for_damd_fix.json', 'r', encoding='utf-8'))
    goal_list=[]
    for dial_id, dial in data.items():
        goal=dial['goal']
        goal_list.append(goal)
    json.dump(goal_list, open('analysis/goals.json', 'w'), indent=2)

def prepare_for_std_eval(path=None, data=None):
    if path:
        data=json.load(open(path, 'r', encoding='utf-8'))
    new_data={}
    dials=evaluator.pack_dial(data)
    for dial_id in dials:
        new_data[dial_id]=[]
        dial=dials[dial_id]
        for turn in dial:
            if turn['user']=='':
                continue
            entry={}
            entry['response']=turn['resp_gen']
            entry['state']=reader.bspan_to_constraint_dict(turn['bspn_gen'])
            new_data[dial_id].append(entry)
    if path:
        new_path=path[:-5]+'std.json'
        json.dump(new_data, open(new_path, 'w'), indent=2)
    return new_data

if __name__=='__main__':
    #path1='/home/liuhong/myworkspace/experiments_21/all_turn-level-DS-11-26_sd11_lr0.0001_bs8_ga4/best_score_model/result.json'
    #prepare_for_std_eval(path1)
    variance_compare()
    #path2='RL_exp/rl-10-19-use-scheduler/best_DS/result.json'
    #unseen_turns=find_unseen_sys_act()
    #calculate_unseen_acc(unseen_turns, path1, path2)
    #compare_offline_result(path1, path2, show_num=30)
    #path1='experiments_21/turn-level-DS/best_score_model/validate_result.json'
    #path2='RL_exp/rl-10-19-use-scheduler/best_DS/validate_result.json'
    #compare_online_result(path1, path2)
    #bspn='[restaurant] pricerange expensive area west'
    #print(reader.bspan_to_DBpointer(bspn, ['restaurant']))
    #unseen_dials=find_unseen_usr_act(path1, path2)
    #print(unseen_dials)
    #act='[taxi] [inform] destination cambridge train station [taxi] [request] car'
    #print(reader.aspan_to_act_dict(act, 'user'))
    #print(set(reader.aspan_to_act_dict(act, 'user')))
    #extract_goal()
