import time

import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import Dataset , DataLoader
from tqdm import tqdm
import math
import copy
from torch.utils.tensorboard import SummaryWriter
import models


import data_loader


#function for saving model
def save_model(mn,best_bert , best_classifier , config,scl_loss,acc,f1,emotion_dict):

    model_name = ['weight.%.2f' % config.pair_weight_exps[mn[1]], '_%s_' %config.weight_where[mn[0]] ,
                  'T.%.2f' % config.temperature,'scl_lr.%s' % str(config.scl_lr),
                  'fine_lr.%s' % str(config.finetuning_lr),'beta.%.2f' % config.beta ,'frozen' if config.bert_freezing==1 else 'unfrozen' ,
                  'curl_epoch' if config.curl_epoch==1 else 'curl_iter',
                  'scl_loss.%.3f' %scl_loss ,'dcrate.%.2f' %config.decrease_curl_rate , 'acc.%.2f' % (acc * 100) ,'f1.%.2f' % (f1 * 100)]

    model_name ='./models/' + '_'.join(model_name) + '.pth'
    print(model_name)

    models = {
        'bert' : best_bert.state_dict(),
        'classifier' : best_classifier.state_dict(),
        'config' : config,
        'emotion_dict' : emotion_dict
    }
    torch.save(models, model_name )


def get_log_name(config,nth):

    file_name = ''

    file_name += '%s - ' % config.train_data
    file_name += 'weight where : %s - ' % (config.weight_where)
    file_name += 'weight: %.2f - ' % (config.pair_weight_exps)[nth]
    file_name += 'scl_lr : %s - ' % str(config.scl_lr)
    file_name += 'fine_lr : %s -' % str(config.finetuning_lr)
    if config.only_best_check:
        file_name += ' %s - ' % config.only_best_check
    else:
        file_name += ' last_save - '

    return file_name


def train(mn,config,bert,classifier, bert_optimizer,classify_optimizer , cross_entropy_crit,
                                      SCL_crit,train_loader,valid_loader):

    if config.use_tensorboard:
        writer = SummaryWriter(config.log_path , get_log_name(config,mn[1]))
        writer.add_text('args',str(config))

    curriculum_rate = 0
    best_scl_loss = 9999.0
    best_bert = None
    early_stop_check = 0

    #train bert for supervised contrastive learning first
    if config.only_cross_entropy == False:
        for epoch in range(config.n_epochs):

            bert.train()
            classifier.train()

            if config.curl_epoch:   #curriculum learing for epoch step
                if config.decrease_curl_rate > 0 and epoch>0:
                    curriculum_rate += config.decrease_curl_rate
                elif config.decrease_curl_rate == 0:
                    curriculum_rate = -1
            else:                   #curriculum learning for iteration step
                curriculum_rate = 0
            train_len = len(train_loader)

            for i, batch_data in tqdm(enumerate(train_loader)):

                if config.curl_epoch==False:
                   curriculum_rate += 1/train_len

                sentences,labels = batch_data
                for param in bert.parameters():
                    param.requires_grad = True
                u_cls = bert(sentences)  # (batch size, hidden size)
                SCL_loss =  SCL_crit(u_cls , labels,curriculum_rate)   # supervised contrastive learning
                bert_optimizer.zero_grad()
                SCL_loss.backward()
                bert_optimizer.step()


            total_acc = 0
            losses = 0
            scl_losses = 0
            nows = 0
            datanum = 0
            y_preds , y_trues = torch.tensor([]).to(config.device) , torch.tensor([]).to(config.device)
            bert.eval()
            classifier.eval()
            with torch.no_grad():   #validation
                for i, batch_data in tqdm(enumerate(valid_loader)):
                    sentences , labels = batch_data
                    u_cls = bert(sentences)
                    classes = classifier(u_cls)

                    scl_loss = SCL_crit(u_cls,labels)
                    scl_losses += scl_loss * len(labels)
                    datanum += len(labels)
                    loss = cross_entropy_crit(classes, labels)
                    losses += loss
                    acc = (torch.argmax(classes,dim=-1) == labels).sum() / len(labels)
                    total_acc += acc

                    y_preds = torch.cat([y_preds,classes],dim=0)
                    y_trues = torch.cat([y_trues,labels],dim=0)

                    nows += 1

                f1_score = models.f1_score(y_preds,y_trues)

            print(f'{epoch + 1}epoch contrastive loss : {scl_losses / datanum}')
            print(f'{epoch + 1}epoch finetuning loss : {losses / nows}')
            print(f'{epoch + 1}epoch acc : {total_acc / nows}')
            print(f'{epoch + 1}epoch f1 : {f1_score}')

            if config.only_best_check:
                if  min(best_scl_loss,scl_losses / datanum) == scl_losses / datanum:
                    print('best check!')
                    best_scl_loss = scl_losses / datanum
                    best_bert = copy.deepcopy(bert)
                    early_stop_check = 0
            else : ##last model save
                print('last save!')
                best_bert = copy.deepcopy(bert)

            if config.use_tensorboard:
                writer.add_scalar('scl epoch',epoch)
                writer.add_scalar('val loss', scl_losses/datanum )

            early_stop_check += 1
            if early_stop_check >= config.early_stopping and config.only_best_check:
                print(f'early stop!!')
                break



    #fine tune
    best_acc , best_f1= torch.tensor(0.0).cuda(config.device) , torch.tensor(0.0).cuda(config.device)
    best_classifier = None
    early_stop_check = 0
    if config.only_cross_entropy:
        pass
    else:
        bert = best_bert.cuda(config.device)

    for epoch in range(config.finetune_epochs):

        y_preds, y_trues = torch.tensor([]).to(config.device), torch.tensor([]).to(config.device)
        classifier.train()
        bert.train()
        for i, batch_data in tqdm(enumerate(train_loader)):

            sentences, labels = batch_data

            if config.bert_freezing :       #option for bert freezing when finetuine
                for param in bert.parameters():
                    param.requires_grad = False

                with torch.no_grad():
                    u_cls = bert(sentences)  # (batch size, hidden size)
            else:
                u_cls = bert(sentences)
                bert_optimizer.zero_grad()

            classes = classifier(u_cls)   # (batch size , labels)
            ce_loss =  config.ce_alpha * cross_entropy_crit(classes, labels)

            classify_optimizer.zero_grad()
            ce_loss.backward()
            classify_optimizer.step()

            if config.bert_freezing == False:
                bert_optimizer.step()

        scl_losses = 0
        total_acc = 0
        acc = 0
        losses = 0
        nows = 0
        bert.eval()
        classifier.eval()
        correct = 0
        datanum  = 0
        with torch.no_grad():       #validation
            for i, batch_data in tqdm(enumerate(valid_loader)):
                sentences , labels = batch_data
                u_cls = bert(sentences)
                classes = classifier(u_cls)
                if config.only_cross_entropy:
                    scl_loss = 999
                else:
                    scl_loss = SCL_crit(u_cls,labels)
                scl_losses += scl_loss
                loss = cross_entropy_crit(classes, labels)
                losses += loss
                correct += (torch.argmax(classes,dim=-1) == labels).sum()
                datanum += len(labels)
                total_acc += acc

                y_preds = torch.cat([y_preds, classes], dim=0)
                y_trues = torch.cat([y_trues, labels], dim=0)

                nows += 1

            acc = correct/datanum
            f1_score = models.f1_score(y_preds, y_trues)

        #different save model standard for best_check option
        if config.best_check == 'acc':
            if torch.max(best_acc,acc) == acc :
                best_bert = copy.deepcopy(bert)
                best_classifier = copy.deepcopy(classifier)
                best_f1 = f1_score
                best_acc = acc
                early_stop_check = 0
                print(f'best acc: {best_acc}')
        elif config.best_check == 'f1':
            if torch.max(best_f1,f1_score) == f1_score :
                best_bert = copy.deepcopy(bert)
                best_classifier = copy.deepcopy(classifier)
                best_f1 = f1_score
                best_acc = acc
                early_stop_check = 0
                print(f'best f1: {best_f1}')

        print(f'finetune {epoch + 1}epoch contrastive loss : {scl_losses/nows}')
        print(f'finetune {epoch + 1}epoch finetuning loss : {losses / nows}')
        print(f'finetune {epoch + 1}epoch acc : {correct/datanum}')
        print(f'finetune {epoch + 1}epoch f1 : {f1_score}')

        if config.use_tensorboard:
            writer.add_scalar('finuetune epoch',epoch)
            writer.add_scalar('finetune ce loss' , losses/nows)
            writer.add_scalar('val acc',correct/datanum)
            writer.add_scalar('f1', f1_score)

        early_stop_check += 1
        if early_stop_check >= config.finetuning_early_multiply * config.early_stopping:
            print('early stop!')
            break

    writer.close()

    save_model(mn,best_bert,best_classifier,config,best_scl_loss,best_acc,best_f1,data_loader.emotion_dict)



