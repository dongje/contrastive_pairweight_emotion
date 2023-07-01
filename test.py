
import torch.nn as nn
import torch
from torch.utils.data import Dataset , DataLoader
from tqdm import tqdm
import argparse
import data_loader
import models



def define_arguparser():

    p = argparse.ArgumentParser()

    p.add_argument('--load_model_name' , required=True)
    p.add_argument('--device',required=True,type=int )
    p.add_argument('--test_data', required=True)
    p.add_argument('--batch_size',type=int , default = 64)


    config = p.parse_args()

    return config


def test(bert , classifier , config  , test_loader):

    cross_entropy_crit = nn.NLLLoss(reduction='mean')

    bert.eval()
    classifier.eval()
    losses = 0
    total_acc = 0
    nows = 0
    datanum = 0
    correct = 0
    y_preds, y_trues = torch.tensor([]).to(config.device), torch.tensor([]).to(config.device)
    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(test_loader)):
            sentences, labels = batch_data
            u_cls = bert(sentences)
            classes = classifier(u_cls)

            loss = cross_entropy_crit(classes, labels)
            losses += loss
            acc = (torch.argmax(classes, dim=-1) == labels).sum() / len(labels)
            total_acc += acc
            correct += (torch.argmax(classes, dim=-1) == labels).sum()
            datanum += len(labels)
            y_preds = torch.cat([y_preds, classes], dim=0)
            y_trues = torch.cat([y_trues, labels], dim=0)

            nows += 1

        f1_score = models.f1_score(y_preds, y_trues)

    models.acc_each_class(y_preds, y_trues)

    print(f'ce loss : {"%.2f" %(losses / nows)}')
    print(f'total acc {"%.2f" % (correct/datanum * 100)}')
    print(f'f1 score : {"%.2f" % (f1_score * 100)}')




def main(config):

    #data and parameter load
    saved_data = torch.load(
        config.load_model_name,
        map_location='cpu' if config.device < 0 else 'cuda:%d' % config.device
    )

    trained_config = saved_data['config']
    saved_classifier = saved_data['classifier']
    saved_bert = saved_data['bert']
    emotion_dict = saved_data['emotion_dict']

    bert = models.emotion_bert(768,trained_config.lang)
    classifier = models.classifier(768,len(emotion_dict))

    classifier.load_state_dict(saved_classifier)
    bert.load_state_dict(saved_bert)

    test_set = data_loader.emotion_bert_dataset(config.test_data, bert.tokenizer, config.device, trained_config.max_length,
                                                 load_emotion_dict=emotion_dict,train=False)
    test_loader =  DataLoader(test_set , batch_size=config.batch_size , shuffle = True)

    print(config.test_data)
    print(trained_config)

    if config.device >=0:
        bert.cuda(config.device)
        classifier.cuda(config.device)

    test(bert,classifier,config,test_loader)


if __name__ == '__main__':
    config = define_arguparser()
    main(config)
