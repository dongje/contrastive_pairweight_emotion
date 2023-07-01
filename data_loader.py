

import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import Dataset , DataLoader


emotion_dict = {}
label_emotion_dict = {}

def get_value(key):
    return emotion_dict.get(key)


class emotion_bert_dataset(Dataset):

    def __init__(self,path,tokenizer,device,max_length,load_emotion_dict= None,train=True,sep = ','):
        df = pd.read_csv(path , sep = sep)

#        df = df.sample(frac=1).reset_index(drop=True)
        self.tokenizer = tokenizer

        self.sentences , self.labels= [] , []
        print('reading')

        self.sentences = tokenizer(df['sentences'][:].to_list() ,max_length=max_length,padding='max_length',truncation = True,return_tensors='pt').to(device)
#        print(self.sentences)
        emotions = df['emotion'][:].to_list()
        if train:
            global emotion_dict
            list_emotion = list(set(emotions))
            for i, emotion in enumerate(list_emotion):
                emotion_dict[emotion] = i
                label_emotion_dict[i] = emotion

        if load_emotion_dict is not None:
            emotion_dict = load_emotion_dict
            for key in emotion_dict.keys():
                label_emotion_dict[emotion_dict.get(key)] = key

        labels = list(map(lambda x:emotion_dict.get(x),emotions))
        self.labels = torch.tensor(labels).to(device)
#        print(self.labels)

    def __len__(self):
        return len(self.sentences['input_ids'])

    def __getitem__(self, idx):
        sentences = {}
        for key in self.sentences.keys():
            sentences[key] =  self.sentences[key][idx]
        return sentences , self.labels[idx]


#emotion_bert_dataset('dataset/1dialog_kor/1dialog_kor_train.csv',1,0,64)









