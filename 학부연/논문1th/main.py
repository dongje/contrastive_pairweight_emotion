

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset , DataLoader
from tqdm import tqdm
import argparse
from torch import optim
import data_loader
import train
import models
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random


def define_arguparser():

    p = argparse.ArgumentParser()

    p.add_argument('--device',required=True,type=int ,help='cuda:#  input -1 if cpu')

    p.add_argument('--train_data', required=True)

    p.add_argument('--valid_data', required=True)

    p.add_argument('--n_epochs', type=int,default=15)

    p.add_argument('--finetune_epochs', type=int, default=30)

    p.add_argument('--max_length',type=int , default=128)

    p.add_argument('--batch_size',type = int,default = 32)

    p.add_argument(
        '--bert_freezing' ,
       type = int ,
       default = 1 ,
       help='if its true , bert is frozen when finetuning',
    )  ## 1 = bert freeze , 0 = not freeze

    p.add_argument(
        '--only_cross_entropy',
        action='store_true',
        help='not use contrastive learning loss but only cross entropy',
    )

    p.add_argument(
        '--only_best_check',
        action='store_true',
        help='if it is true , we save only best model on plm ',
    )

    p.add_argument(
        '--regular',
        action='store_true',
        help='positive or negative regularize',
    )

    p.add_argument(
        '--use_tensorboard',
        action='store_true'
    )

    p.add_argument(
        '--curl_epoch',
        action='store_true'
    )

    p.add_argument(
        '--log_path',
        type = str ,
        default='logs/'
    )

    p.add_argument(
        '--use_random_seed',
        action= 'store_true',
        )

    p.add_argument('--random_seed',type = int,default = 2023)

    p.add_argument('--early_stopping' , type = int , default=8)

    p.add_argument('--finetuning_early_multiply',
                   type=int,
                   default=2 ,
                   help='when finetune , early stop is early_stop * this value (ex. 8*2)'
                   )

    p.add_argument('--lang',type=str,default='en' , help='kr or en')

    p.add_argument('--scl_lr',
                   type=float,
                   default=1e-5 ,
                   help= ' learning rate for contrastive learning. it can be different from finetuing lr'
                   )

    p.add_argument('--finetuning_lr',type=float,default=1e-5 )

    p.add_argument('--pair_weight_exps',nargs='+',
                   type=float,
                   default = [1.0],
                   help = 'by the cosine similarity , torch.pow(cosine) will be weighted if similar , weighted more. if weight is 1 , there is no weight. it can get many things')

    p.add_argument('--decrease_curl_rate' ,
                   type = float ,
                   default = 0,
                   help= 'if weight on positive pair , you can do curriculum learning. the rate is diminishing unit when training.  curriculum rate decreases until 0')

    p.add_argument('--weight_where' ,
                   nargs = '+',
                   type = str ,
                   default = ['nagative'] ,  ## plus  cosine
                    help='if positive , it will be softlabel for resemble , if negative , the loss will be punished more for hard negative. just set pair_weight_exp = 1 for not weight anything')

    p.add_argument('--beta' , type = float , default = 1.0 , help = 'the importance scale for concantrating on pos or neg pair. 1 is lowest. ')

    p.add_argument('--ce_alpha' , type=float , default = 1.0 , help = 'recommend do not change')


    p.add_argument('--temperature' , type=float , default= 0.1 , help = 'temperature parameter for similarity sharpness')

    p.add_argument('--best_check', type=str, default='acc', help='acc or f1')

    config = p.parse_args()


    assert config.best_check == 'acc' or config.best_check == 'f1'

    if config.bert_freezing == 1:
        if config.only_cross_entropy:
            config.bert_freezing = False
        else:
            config.bert_freezing = True
    else:
        config.bert_freezing = False
    print(config)
    if config.use_random_seed:
        set_random_seed(config.random_seed)

    return config



def get_crit(use = 'contrastive'):

    crit = nn.NLLLoss( reduction='mean')
    return crit

def set_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_double_optimizer(bert,classifier, config):
    bert_optimizer = optim.Adam(bert.parameters(), lr=config.scl_lr, betas=(.9, .98),eps=1e-9)
    finetuning_optimizer = optim.Adam(classifier.parameters(), lr=config.finetuning_lr, betas=(.9, .98),eps=1e-9)
#    params = list(bert.parameters()) + list(classifier.parameters())
#    optimizer = optim.Adam(params, lr=config.lr, betas=(.9, .98),eps=1e-9)
    return bert_optimizer,finetuning_optimizer


def main(config):

    print(config.pair_weight_exps)

    for m , weight_where in enumerate(config.weight_where):

        for n,pair_weight_exp in enumerate(config.pair_weight_exps):

            print(config)
            print(f'{m} {weight_where} , {n+1}th weight_exp : {pair_weight_exp}')

            bert = models.emotion_bert(768,lang=config.lang)

            SCL_crit = models.SCLloss(config , pair_weight_exp,weight_where)
            cross_entropy_crit = get_crit('c.e')

            train_set = data_loader.emotion_bert_dataset(config.train_data , bert.tokenizer , config.device , config.max_length  ,train=True)
            valid_set = data_loader.emotion_bert_dataset(config.valid_data, bert.tokenizer, config.device, config.max_length , train=False)

            emotion_classifier = models.classifier(768,len(data_loader.emotion_dict))
            bert_optimizer , classify_optimizer = get_double_optimizer(bert,emotion_classifier,config)


            train_loader = DataLoader(train_set , batch_size=config.batch_size , shuffle = True)
            valid_loader = DataLoader(valid_set,batch_size=config.batch_size , shuffle = True)

            if config.device >= 0 :
                bert.cuda(config.device)
                emotion_classifier.cuda(config.device)
                SCL_crit.cuda(config.device)
                cross_entropy_crit.cuda(config.device)
            else:
                device = torch.device('cpu')


            train.train([m,n],config,bert,emotion_classifier,bert_optimizer , classify_optimizer
                        ,cross_entropy_crit,SCL_crit,train_loader,valid_loader)


if __name__ == '__main__':
    config = define_arguparser()
    main(config)











