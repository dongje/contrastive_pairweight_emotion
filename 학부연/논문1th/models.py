
import torch.nn as nn
import torch.nn.functional as F
import torch
import data_loader
from transformers import BertTokenizer,BertModel, EncoderDecoderModel,BertTokenizerFast,AutoModel,AutoTokenizer
import math

#the degree of emotion about 2-dimensional valence and arousal model
def get_emotion_degree(emotion):
    pie = math.pi

    if emotion == 'happy':
        return pie/10
    elif emotion =='surprise':
        return 5*pie/12
    elif emotion =='angry':
        return 7*pie/12
    elif emotion =='anger':
        return 7*pie/12
    elif emotion =='disgust':
        return 3*pie/4
    elif emotion =='fear':
        return 11*pie/12
    elif emotion =='frustrated':
        return 13*pie/12
    elif emotion =='sad':
        return 16*pie/12
    elif emotion == 'joy':  ## .. is it correct?   before 2023 4 19 10:30 ,, joy was pie/4
        return 1 * pie / 3
    elif emotion =='powerful':
        return 1* pie / 4
    elif emotion =='peaceful':
        return 23 * pie / 12
    elif emotion =='scared':
        return 2 * pie / 3
    elif emotion=='neutral':
        return 0

#angle between two emotions
def get_valent_zero_mask(emo1_pies,emo2_pies):

    mul_cosine = torch.cos(emo1_pies) * torch.cos(emo2_pies)

    return (mul_cosine < 0)

#vanilla negative weight
def weight_nagative_emotion(emotion_label1 , emotion_label2,a,curriculum_rate=0):  ## label1 anchor tensor (batch , batch) label2 transposed tnesor for same size

    emotion_label1 = emotion_label1.to('cpu')
    emotion_label2 = emotion_label2.to('cpu')
    emotion_label1 = emotion_label1.tolist()
    emotion_label2 = emotion_label2.tolist()

    emotion1 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label1]
    emotion2 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label2]
 #   print(emotion1[:2])
    emo1_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion1]).to(0)
    emo2_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion2]).to(0)

    equal_mask = torch.eq(emo1_pies , emo1_pies.T)
    neutral_mask = ((emo1_pies == 0).bool() + (emo2_pies == 0).bool())
    valent_zero_mask = get_valent_zero_mask(emo1_pies,emo2_pies)
    zero_mask =(~equal_mask * ~neutral_mask ).int()  ## value 0 at neutral or samething relation

    half_pie_neutral_mask = ( equal_mask + neutral_mask ).int() * math.pi / 2  #place neutral pie/2 and others 0

    abs_pies = torch.abs((emo1_pies - emo2_pies))

    cos = torch.cos((abs_pies * zero_mask) + half_pie_neutral_mask)  #make neutral pie/2 and cos.. 0
    cos[cos<0] = 0

    powed = torch.pow(a,cos)
    if curriculum_rate == -1:   #no curriculum
        return powed
    else:       #curriculum
        return torch.pow(powed,(curriculum_rate))

# cos(a + r * b) b is the similarity between labels
def negative_consine(emotion_label1 , emotion_label2,a,temperature,curriculum_rate):
    emotion_label1 = emotion_label1.to('cpu')
    emotion_label2 = emotion_label2.to('cpu')
    emotion_label1 = emotion_label1.tolist()
    emotion_label2 = emotion_label2.tolist()

    emotion1 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label1]
    emotion2 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label2]
    #   print(emotion1[:2])
    emo1_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion1]).to(0)
    emo2_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion2]).to(0)

    equal_mask = torch.eq(emo1_pies, emo1_pies.T)
    neutral_mask = ((emo1_pies == 0).bool() + (emo2_pies == 0).bool())
    valent_zero_mask = get_valent_zero_mask(emo1_pies,emo2_pies)
    zero_mask = (~equal_mask * ~neutral_mask ).int()  ## value 0 at neutral or samething relation

    half_pie_neutral_mask = (equal_mask + neutral_mask ).int() * math.pi / 2  # place neutral pie/2 and others 0

    abs_pies = torch.abs((emo1_pies - emo2_pies))

    cos = torch.cos((abs_pies * zero_mask) + half_pie_neutral_mask)  # make neutral pie/2 and cos.. 0
    cos[cos < 0] = 0

    if curriculum_rate == -1:
        powed = torch.pow(a , -cos/temperature)
    else:
        powed = torch.pow(a, -cos/temperature * (1-curriculum_rate) )
    return powed


#
def real_negative_cosine(emotion_label1 , emotion_label2,a,temperature,curriculum_rate):
    emotion_label1 = emotion_label1.to('cpu')
    emotion_label2 = emotion_label2.to('cpu')
    emotion_label1 = emotion_label1.tolist()
    emotion_label2 = emotion_label2.tolist()

    emotion1 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label1]
    emotion2 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label2]
    #   print(emotion1[:2])
    emo1_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion1]).to(0)
    emo2_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion2]).to(0)

    equal_mask = torch.eq(emo1_pies, emo1_pies.T)
    neutral_mask = ((emo1_pies == 0).bool() + (emo2_pies == 0).bool())
    valent_zero_mask = get_valent_zero_mask(emo1_pies,emo2_pies)
    zero_mask = (~equal_mask * ~neutral_mask ).int()  ## value 0 at neutral or samething relation

    half_pie_neutral_mask = (equal_mask + neutral_mask ).int() * math.pi / 2  # place neutral pie/2 and others 0

    abs_pies = torch.abs((emo1_pies - emo2_pies))

    cos = torch.cos((abs_pies * zero_mask) + half_pie_neutral_mask)  # make neutral pie/2 and cos.. 0
    cos[cos < 0] = 0

    if curriculum_rate == -1:
        powed = torch.pow(a , cos/temperature)
    else:
        powed = torch.pow(a, cos/temperature * (1-curriculum_rate) )
    return powed

#weight only cosb. b is angle between emotions
def weight_positive_emotion(emotion_label1 , emotion_label2,a):
    emotion_label1 = emotion_label1.to('cpu')
    emotion_label2 = emotion_label2.to('cpu')
    emotion_label1 = emotion_label1.tolist()
    emotion_label2 = emotion_label2.tolist()

    emotion1 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label1]
    emotion2 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label2]

    emo1_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion1]).to(0)
    emo2_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion2]).to(0)

    equal_mask = torch.eq(emo1_pies, emo1_pies.T)
    neutral_mask = ((emo1_pies == 0).bool() + (emo2_pies == 0).bool())
    zero_mask = (~equal_mask * ~neutral_mask).int()  ## value 0 at neutral or samething relation
    half_pie_neutral_mask = (equal_mask + neutral_mask).int() * math.pi / 2  # place neutral pie/2 and others 0

    abs_pies = torch.abs((emo1_pies - emo2_pies))

    cos = torch.cos((abs_pies * zero_mask) + half_pie_neutral_mask)  # make neutral pie/2 and cos.. 0
    cos[cos < 0] = 0

    powed_ = torch.pow(a,cos) / a

    return powed_ * (zero_mask) * (cos > 0).int()  ## mask on positive pair

#weight plus cosb on vector similarity cosa = cos(a+b)
def plus_cosine(emotion_label1 , emotion_label2,a,temperature,curriculum_rate):
    emotion_label1 = emotion_label1.to('cpu')
    emotion_label2 = emotion_label2.to('cpu')
    emotion_label1 = emotion_label1.tolist()
    emotion_label2 = emotion_label2.tolist()

    emotion1 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label1]
    emotion2 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label2]

    emo1_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion1]).to(0)
    emo2_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion2]).to(0)

    equal_mask = torch.eq(emo1_pies, emo1_pies.T)
    neutral_mask = ((emo1_pies == 0).bool() + (emo2_pies == 0).bool())
    valent_zero_mask = get_valent_zero_mask(emo1_pies,emo2_pies)
    zero_mask = (~(equal_mask + neutral_mask) ).int()  ## value 0 at neutral or samething relation or valent opposite
    half_pie_neutral_mask = (equal_mask + neutral_mask).int() * math.pi / 2  # place neutral pie/2 and others 0

    abs_pies = torch.abs((emo1_pies - emo2_pies))
    abs_pies[abs_pies > math.pi / 2] = math.pi/2
    cos = torch.cos((abs_pies * zero_mask) + half_pie_neutral_mask)  # make neutral pie/2 and cos.. 0
    cos[cos < 0] = 0

    return cos

#weight a^(cosa - cosb_
def plus_real_cosine(emotion_label1 , emotion_label2,a,temperature,curriculum_rate):
    emotion_label1 = emotion_label1.to('cpu')
    emotion_label2 = emotion_label2.to('cpu')
    emotion_label1 = emotion_label1.tolist()
    emotion_label2 = emotion_label2.tolist()

    emotion1 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label1]
    emotion2 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label2]

    emo1_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion1]).to(0)
    emo2_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion2]).to(0)

    equal_mask = torch.eq(emo1_pies, emo1_pies.T)
    neutral_mask = ((emo1_pies == 0).bool() + (emo2_pies == 0).bool())
    valent_zero_mask = get_valent_zero_mask(emo1_pies,emo2_pies)
    zero_mask = (~(equal_mask + neutral_mask )).int()  ## value 0 at neutral or samething relation


    half_pie_neutral_mask = (equal_mask + neutral_mask ).int() * math.pi / 2  # place neutral pie/2 and others 0

    abs_pies = torch.abs((emo1_pies - emo2_pies))
    abs_pies[abs_pies > math.pi / 2] = math.pi/2
    cos = torch.cos((abs_pies * zero_mask) + half_pie_neutral_mask)  # make neutral pie/2 and cos.. 0
    cos[cos < 0] = 0
    powed = torch.pow(a,-cos/temperature)
    powed[powed ==1] = 0
    return powed

#input sin(a-b) rather than cos(a+b)
def minus_sin(emotion_label1 , emotion_label2,a,temperature,curriculum_rate):
    emotion_label1 = emotion_label1.to('cpu')
    emotion_label2 = emotion_label2.to('cpu')
    emotion_label1 = emotion_label1.tolist()
    emotion_label2 = emotion_label2.tolist()

    emotion1 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label1]
    emotion2 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label2]
    #   print(emotion1[:2])
    emo1_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion1]).to(0)
    emo2_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion2]).to(0)

    equal_mask = torch.eq(emo1_pies, emo1_pies.T)
    neutral_mask = ((emo1_pies == 0).bool() + (emo2_pies == 0).bool())
    valent_zero_mask = get_valent_zero_mask(emo1_pies,emo2_pies)
    live_mask = (~equal_mask * ~neutral_mask * ~valent_zero_mask).int()  ## value 0 at neutral or samething relation

    half_pie_neutral_mask = (equal_mask + neutral_mask).int() * math.pi / 2  # place neutral pie/2 and others 0

    abs_pies = torch.abs((emo1_pies - emo2_pies))

    sin = torch.sin((abs_pies * live_mask) )  # make neutral ,equal , valent opposite 0

    if curriculum_rate == -1:
        powed = torch.pow(a , -sin/temperature)
    else:
        powed = torch.pow(a, -sin/temperature )
    return powed

#positive weight cos(a+b)  b is similarity
def summation_cosine(emotion_label1 , emotion_label2,a,temperature,curriculum_rate):
    emotion_label1 = emotion_label1.to('cpu')
    emotion_label2 = emotion_label2.to('cpu')
    emotion_label1 = emotion_label1.tolist()
    emotion_label2 = emotion_label2.tolist()

    emotion1 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label1]
    emotion2 = [[data_loader.label_emotion_dict.get(x) for x in row] for row in emotion_label2]

    emo1_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion1]).to(0)
    emo2_pies = torch.tensor([[get_emotion_degree(x) for x in row] for row in emotion2]).to(0)

    equal_mask = torch.eq(emo1_pies, emo1_pies.T)
    neutral_mask = ((emo1_pies == 0).bool() + (emo2_pies == 0).bool())
    zero_mask = (~equal_mask * ~neutral_mask).int()  ## value 0 at neutral or samething relation

    abs_pies = torch.abs((emo1_pies - emo2_pies))
    abs_pies[abs_pies > math.pi / 2] = 0
    return abs_pies * zero_mask  ## should plus theta

#bert and classifier
class emotion_bert(nn.Module):

    def __init__(self,hidden_size,lang='en'):
        super().__init__()
        if lang == 'en':
            model_name = 'bert-base-uncased'
            self.base_bert = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        elif lang =='kr':
            model_name = 'klue/bert-base'
            self.base_bert = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self,sentence ):          ## u , v ==>> | 1 , hidden |
        u = self.base_bert(**sentence)[0]
        u_cls = u[:,0,:]
        return u_cls

#hidden x classes
class classifier(nn.Module):

    def __init__(self,hidden_size,num_class):
        super().__init__()
        self.class_matrix = nn.Linear(hidden_size ,num_class)

    def forward(self,u_cls ):          ## u , v ==>> | 1 , hidden |
        results = self.class_matrix(u_cls)
        return  torch.log_softmax(results,dim=-1)

#supervised contrastive learning loss function
class SCLloss(nn.Module):

    def __init__(self,config,pair_weight_exp,weight_where):
        super().__init__()
        assert pair_weight_exp > 0
        self.temperature = config.temperature
        self.device = config.device
        self.weight_where = weight_where
        self.regular = config.regular
        self.pair_weight_exp = pair_weight_exp
        self.Beta = config.beta

    def forward(self,features,labels,curriculum_rate = -1): # (batch size , hidden) , (batch size , 1)

        version = 'nonmoon'

        ## i once implemented sup contrstive learning for my paper version but it performed not good at well. i'll find the better implement at the future
        if version == 'mine':

            total_loss = torch.tensor(0.0).cuda(self.device).clone()
            features_dict = {}
            negative_sim = torch.tensor(0.0).cuda(self.device).clone()
            for label in data_loader.label_emotion_dict.keys():
                label_mask = (labels == label)
                features_dict[label] = features[label_mask]   ## split features by class

            for anchor_label in features_dict.keys():
                positive_tensor = features_dict[anchor_label].clone()
                for i in range(positive_tensor.size(0)):
                    anchor_tensor = positive_tensor[i:i+1].clone()

                    for pair_label in features_dict.keys():
                        pair_tensor = features_dict[pair_label]
                        if pair_label == anchor_label:     ##positive pair
                            if pair_tensor.size(0) <= 1:
                                pos_exp_x = None
                                continue
                            excluded_positve = torch.cat((pair_tensor[:i],pair_tensor[i+1:]),dim=0)
                            anchor = anchor_tensor.expand(excluded_positve.size(0),-1)
                            cosine_sim = F.cosine_similarity(anchor, excluded_positve) / self.temperature
                            pos_exp_x = torch.exp(cosine_sim)
                            negative_sim = negative_sim.clone() + torch.sum(pos_exp_x,dim=0)

                        else:        ##negative pair
                            if pair_tensor.size(0) == 0:
                                continue
                            anchor = anchor_tensor.expand(pair_tensor.size(0),-1)
                            cosine_sim = F.cosine_similarity(anchor,pair_tensor) / self.temperature
                            neg_exp_x = torch.exp(cosine_sim)
                            negative_sim = negative_sim.clone() + torch.sum(neg_exp_x,dim=0)

                    loss = torch.tensor(0.0).cuda(self.device).clone()
                    if torch.is_tensor(pos_exp_x):  ##positive pair plus
                        for j in range(pos_exp_x.size(0)):
                            positive_sim = pos_exp_x[j].clone()
    #                        print(positive_sim)
    #                        print(negative_sim)
                            loss.add_(torch.log(positive_sim / negative_sim))
                        loss = loss.clone() * -1 / (pos_exp_x.size(0))
                        total_loss = total_loss.clone() + loss

            return total_loss / labels.size(0)

        ## it was revised based on the 2020 supervised contrastiver learning paper github code.
        ## the original for 3d data but we input only 2d data ex.(batch size , hidden) .. so i revised somethings

        elif version == 'nonmoon':

            device = (torch.device('cuda')
                      if features.is_cuda
                      else torch.device('cpu'))
            if curriculum_rate > 1:
                curriculum_rate = 1
            batch_size = features.shape[0]

            expanded_labels = labels.clone().unsqueeze(0).expand(batch_size, -1)
            if self.weight_where == 'negative' and self.pair_weight_exp>=1:
                nagative_weight_mask = weight_nagative_emotion(expanded_labels, expanded_labels.T,self.pair_weight_exp,curriculum_rate)  ##weight mask for positive or negative pair along with emotion degree
            elif self.weight_where == 'positive' and self.pair_weight_exp>1:
                positive_weight_mask = weight_positive_emotion(expanded_labels, expanded_labels.T,self.pair_weight_exp)
                sum1_mask = (positive_weight_mask.sum(1) == 0).to(0)
                mask_sum = (positive_weight_mask > 0).int().sum(1) + sum1_mask.int()
                mask_sum = mask_sum.reshape(batch_size, 1)
                positive_weight_mask = positive_weight_mask / mask_sum
            elif self.weight_where == 'cosine' and self.pair_weight_exp > 1:
                positive_weight_mask = plus_cosine(expanded_labels, expanded_labels.T, self.pair_weight_exp,self.temperature,curriculum_rate)
                sum1_mask = (positive_weight_mask.sum(1) == 0).to(0)   #
                mask_sum = (positive_weight_mask > 0).int().sum(1)*self.Beta + sum1_mask.int()
                mask_sum = mask_sum.reshape(batch_size,1)
                positive_weight_mask = positive_weight_mask/mask_sum
            elif self.weight_where == 'real_cosine' and self.pair_weight_exp > 1:
                positive_weight_mask = plus_real_cosine(expanded_labels, expanded_labels.T, self.pair_weight_exp,self.temperature,curriculum_rate)
                sum1_mask = (positive_weight_mask.sum(1) == 0).to(0)
                mask_sum = (positive_weight_mask == 0).int().sum(1)*self.Beta + sum1_mask.int()
                mask_sum = mask_sum.reshape(batch_size,1)
                positive_weight_mask = positive_weight_mask/mask_sum
            elif self.weight_where == 'cosine_negative' and self.pair_weight_exp > 1:
                nagative_weight_mask = real_negative_cosine(expanded_labels, expanded_labels.T, self.pair_weight_exp,self.temperature,curriculum_rate)
            elif self.weight_where =='sum_pos' and self.pair_weight_exp > 1:
                sumed_theta = summation_cosine(expanded_labels, expanded_labels.T, self.pair_weight_exp,self.temperature, curriculum_rate)
            elif self.weight_where == 'sin_pos' and self.pair_weight_exp > 1:
                minus_sin_weight = minus_sin(expanded_labels, expanded_labels.T, self.pair_weight_exp,self.temperature,curriculum_rate)
                one_position = minus_sin_weight == 1
                value_position = minus_sin_weight != 1
                sum1_mask = (minus_sin_weight.sum(1) == batch_size).to(0)
                mask_sum = (minus_sin_weight != 1).int().sum(1) * self.Beta + sum1_mask.int()
                mask_sum = mask_sum.reshape(batch_size,1)
                minus_sin_weight = minus_sin_weight / mask_sum ###  (-sin(a) / T / n)
                minus_sin_weight[one_position] = 1


            elif self.pair_weight_exp == 1:
                positive_weight_mask = torch.zeros(batch_size,batch_size).to(self.device)
            labels = labels.clone().view(batch_size, -1)
            mask = torch.eq(labels, labels.T).float().to(device)

            contrast_count = features.shape[1]
#            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            contrast_mode = 'all'
            if contrast_mode == 'one':
                anchor_feature = features[:, 0]
                anchor_count = 1
            elif contrast_mode == 'all':
                anchor_feature = features
                anchor_count = contrast_count
            else:
                raise ValueError('Unknown mode: {}'.format(contrast_mode))

            normalized_sum = features.norm(dim=-1, keepdim=True)
            features = features / normalized_sum

            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features, features.T),
                self.temperature)
            # for numerical stability
            # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            # logits = anchor_dot_contrast - logits_max.detach()
            logits = anchor_dot_contrast

            # mask-out self-contrast cases
            # logits_mask = torch.scatter(
            #     torch.ones_like(mask),
            #     1,
            #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            #     0
            # )
            logits_mask = torch.ones_like(mask).fill_diagonal_(0).to(device)
            mask = mask #* logits_mask

            if self.weight_where == 'negative' or self.weight_where =='cosine_negative':

                # compute log_prob
                exp_logits = torch.exp(logits) * logits_mask  * nagative_weight_mask
                if self.regular:
                    regular_weight = logits_mask.sum(1) / (logits_mask  * nagative_weight_mask).sum(1)
                else:
                    regular_weight = 1

                log_prob = logits - torch.log(regular_weight * exp_logits.sum(1, keepdim=True))
                sum1_mask = (mask.sum(1) == 1).to(0)
              #  mask_sum = mask.fill_diagonal_(0).sum(1) + sum1_mask.int()
                # compute mean of log-likelihood over positive
                mean_log_prob_pos = (mask * logits_mask * log_prob).sum(1) / mask.sum(1)

            elif self.weight_where == 'positive':
                if curriculum_rate == -1:
                    curriculum_rate = 0
                ## true(except self) mask + positive pair pow(a,(cos-1)) / pair num (for size)
                plus_pos_mask = mask * logits_mask + (positive_weight_mask * (1 - curriculum_rate))

                #preventing for not dividing 0 (not exist) and not mask
                sum1_mask = (plus_pos_mask.sum(1) == 0).to(torch.cuda.current_device())
                final_mask_sum = plus_pos_mask.sum(1) + sum1_mask.int()

                #print((mask*logits_mask).sum(1)[:2])
                # print(final_mask.sum(1)[:2])
#                print(labels[:2])
#                print((positive_weight_mask * (1 - curriculum_rate))[:5])
#                print(torch.div((mask*logits_mask).sum(1) , final_mask.sum(1))[:2])
                if self.regular:
                    if torch.any(final_mask_sum == 0):
                        print(final_mask_sum)
                        print(plus_pos_mask)
                        exit()
                    regular_weight = torch.div((mask*logits_mask).sum(1) , final_mask_sum)
                else:
                    regular_weight = 1

                #compuete  exp(similarity) * positive_weight
                pos_logits = torch.exp(logits) * plus_pos_mask * regular_weight
                pos_logits[pos_logits == 0] = 1

                # compute exponential except for self
                exp_logits = torch.exp(logits) * logits_mask
                log_prob = torch.log(pos_logits) - torch.log(exp_logits.sum(1, keepdim=True))
                if bool(torch.isnan(log_prob).sum()):
#                    print((positive_weight_mask * (1 - curriculum_rate))[:5])
#                    print(torch.div((mask*logits_mask).sum(1) , final_mask.sum(1))[:2])
                    print(plus_pos_mask[:2])
                    print(regular_weight)
                    print(pos_logits[:2])
                    print(log_prob[:2])
#                print(log_prob[:2])
                mean_log_prob_pos = ( log_prob*((mask*logits_mask).bool())).sum(1) / mask.sum(1)
                sum1_mask = (positive_weight_mask.sum(1) == 0).to(torch.cuda.current_device())
                pos_mask_sum = positive_weight_mask.bool().int().sum(1) + sum1_mask.int()
                mean_log_prob_pos += (log_prob*(plus_pos_mask * positive_weight_mask.bool() )* (1-curriculum_rate)).sum(1) / pos_mask_sum

                #for compare , not use
                ll = logits - torch.log(exp_logits.sum(1, keepdim=True))
                mm = (plus_pos_mask * ll).sum(1) / mask.sum(1)

            #pass because the effect will be distracted as i assume
            elif self.weight_where == 'both':
                pass

            #weight cosb between angles
            elif self.weight_where == 'cosine':
                if curriculum_rate == -1:
                    curriculum_rate = 0
                ## true(except self) mask + positive pair pow(a,(cos-1)) / pair num (for size)
                plus_pos_mask = mask * logits_mask + positive_weight_mask*(1-curriculum_rate)

                # preventing for not dividing 0 (not exist) and not mask
                sum1_mask = (plus_pos_mask.sum(1) == 0).to(torch.cuda.current_device())
                final_mask_sum = plus_pos_mask.sum(1) + sum1_mask.int()

                if self.regular:
                    if torch.any(final_mask_sum == 0):
                        print(final_mask_sum)
                        print(plus_pos_mask)
                        exit()
                    regular_weight = torch.div((mask * logits_mask).sum(1), final_mask_sum)
                else:
                    regular_weight = 1

                # compuete  exp(similarity) * positive_weight
                pos_logits = torch.exp(logits) * plus_pos_mask * regular_weight
                pos_logits[pos_logits == 0] = 1

                # compute exponential except for self
                exp_logits = torch.exp(logits) * logits_mask
                log_prob = torch.log(pos_logits) - torch.log(exp_logits.sum(1, keepdim=True))
                if bool(torch.isnan(log_prob).sum()):
                    print(plus_pos_mask[:2])
                    print(regular_weight)
                    print(pos_logits[:2])
                    print(log_prob[:2])
                #                print(log_prob[:2])
                mean_log_prob_pos = (log_prob * ((mask * logits_mask).bool())).sum(1) / mask.sum(1)
                sum1_mask = (positive_weight_mask.sum(1) == 0).to(torch.cuda.current_device())
                pos_mask_sum = positive_weight_mask.bool().int().sum(1) + sum1_mask.int()
                mean_log_prob_pos += (log_prob * (plus_pos_mask * positive_weight_mask.bool())).sum(1) / pos_mask_sum

                # print(labels)
                # print((1-curriculum_rate))
                # print(positive_weight_mask[:2])
                # print(plus_pos_mask[:2])

            #
            elif self.weight_where == 'real_cosine':
                if curriculum_rate == -1:
                    curriculum_rate = 0
                ## true(except self) mask + positive pair pow(a,(cos-1)) / pair num (for size)
                plus_pos_mask = mask * logits_mask + positive_weight_mask * (1 - curriculum_rate)

                # preventing for not dividing 0 (not exist) and not mask
                sum1_mask = (plus_pos_mask.sum(1) == 0).to(torch.cuda.current_device())
                final_mask_sum = plus_pos_mask.sum(1) + sum1_mask.int()

                if self.regular:
                    if torch.any(final_mask_sum == 0):
                        print(final_mask_sum)
                        print(plus_pos_mask)
                        exit()
                    regular_weight = torch.div((mask * logits_mask).sum(1), final_mask_sum)
                else:
                    regular_weight = 1

                # compuete  exp(similarity) * positive_weight
                pos_logits = torch.exp(logits) * plus_pos_mask * regular_weight
                pos_logits[pos_logits == 0] = 1

                # compute exponential except for self
                exp_logits = torch.exp(logits) * logits_mask
                log_prob = torch.log(pos_logits) - torch.log(exp_logits.sum(1, keepdim=True))
                if bool(torch.isnan(log_prob).sum()):
                    print(f'positive weight {positive_weight_mask[:2]}')
                    print(f'plus pos {plus_pos_mask[:2]}')
                    print(f'{regular_weight}')
                    print(f'pos logits {pos_logits[:2]}')
                    print(f'log prob {log_prob[:2]}')
                #                print(log_prob[:2])
                mean_log_prob_pos = (log_prob * ((mask * logits_mask).bool())).sum(1) / mask.sum(1)
                sum1_mask = (positive_weight_mask.sum(1) == 0).to(torch.cuda.current_device())
                pos_mask_sum = positive_weight_mask.bool().int().sum(1) + sum1_mask.int()
                mean_log_prob_pos += (log_prob * (plus_pos_mask * positive_weight_mask.bool())).sum(1) / pos_mask_sum

                # print(positive_weight_mask[:2])
                # print(plus_pos_mask[:2])

            #weight cos(a+b) caculate summation of angles
            elif self.weight_where =='sum_pos':
                if curriculum_rate == -1:
                    curriculum_rate = 0

                pos_mask = sumed_theta.bool().int().to(torch.cuda.current_device())
                cosine_b = torch.cos(sumed_theta)
                sin_b = torch.sin(sumed_theta)
                # theta = torch.acos(torch.matmul(features, features.T))
                cosine_a = torch.matmul(features, features.T)
                cosine_a[cosine_a > 0.9999] = 0.99999   ### 1 to 0.99999 for numerical error because torch.acos(1) = nan
#                sin_square = 1 - torch.square(cosine_a)
#                sin_square[sin_square<0] = 1e-8
#                sin_a = torch.sqrt(sin_square)

                theta_a = torch.acos(cosine_a)
                sin_a = torch.sin(theta_a)
                # theta_gamma = theta  + sumed_theta
                cosine_a_b = cosine_a * cosine_b - sin_a * sin_b

                pos_logits = torch.div(cosine_a_b , self.temperature) * pos_mask  ## have only on similar emotion position

                exp_logits = torch.exp(logits) * logits_mask
                final_logits = logits * (~pos_mask.bool()).int() + pos_logits
                log_prob = final_logits - torch.log(exp_logits.sum(1, keepdim=True))

                mean_log_prob_pos = (log_prob * ((mask * logits_mask).bool())).sum(1) / mask.sum(1)
                sum1_mask = (pos_mask.sum(1) == 0).to(torch.cuda.current_device())
                pos_mask_sum = pos_mask.bool().int().sum(1) + sum1_mask.int()
                mean_log_prob_pos += ( (1-curriculum_rate) *pos_mask.bool().int()* log_prob ).sum(1) / pos_mask_sum

                # print(cosine_a_b[:3])
                # print(pos_logits[:3])
                # print(data_loader.label_emotion_dict)
                # print(labels)
                # print((log_prob * ((mask * logits_mask).bool()).int())[:3])
                # print((pos_mask.bool().int()* log_prob)[:3])
                #
                # print()
                if  bool(torch.isnan(mean_log_prob_pos).sum()):
                    print(features[:3])
                    print(cosine_a[:3])
                    print(torch.acos(cosine_a)[:3])
                    print(sin_a[:3])
                    print(cosine_a_b[:3])
                    print(pos_logits[:3])
                    print(final_logits[:3])
                    print(log_prob[:3])
                    print(exp_logits[:3])
                    print(((mask * logits_mask).bool()))
                    print((log_prob * ((mask * logits_mask).bool().int())).sum(1))
                    print(((1-curriculum_rate) *pos_mask.bool().int()* log_prob  ).sum(1))

                    exit()

            #weight (cosa - sinb) a is similarity and b is angles between emotions
            elif self.weight_where == 'sin_pos':
                if curriculum_rate == -1:
                    curriculum_rate = 0

                def make_eq_and_neut_to_zero(minus_sin_weight_imsi):
                    minus_sin_weight_imsi = minus_sin_weight_imsi.to(torch.cuda.current_device())
                    minus_sin_weight_imsi[minus_sin_weight_imsi == 1e+0] = 0
                    return minus_sin_weight_imsi

                ## true(except self) mask + positive pair pow(e,(cos-1)) / pair num (for size)
                plus_pos_mask = mask * logits_mask + make_eq_and_neut_to_zero(minus_sin_weight.clone()) * (1 - curriculum_rate)


                # preventing for not dividing 0 (not exist) and not mask
                sum1_mask = (plus_pos_mask.sum(1) == 0).to(torch.cuda.current_device())
                final_mask_sum = plus_pos_mask.sum(1) + sum1_mask.int()

                if self.regular:
                    if torch.any(final_mask_sum == 0):
                        print(final_mask_sum)
                        print(plus_pos_mask)
                        exit()
                    regular_weight = torch.div((mask * logits_mask).sum(1), final_mask_sum)
                else:
                    regular_weight = 1

                # compuete  exp(similarity) * positive_weight . done sum
                pos_logits = torch.exp(logits) * plus_pos_mask
                pos_logits[pos_logits == 0] = 1

                # compute exponential except for self
                exp_logits = torch.exp(logits) * logits_mask
                log_prob = torch.log(pos_logits) - torch.log(exp_logits.sum(1, keepdim=True))
                if bool(torch.isnan(log_prob).sum()):
                    print(plus_pos_mask[:2])
                    print(regular_weight)
                    print(pos_logits[:2])
                    print(log_prob[:2])
                #                print(log_prob[:2])
                mean_log_prob_pos = (log_prob * ((mask * logits_mask).bool())).sum(1) / mask.sum(1)
                sum1_mask = (minus_sin_weight.sum(1) == batch_size).to(torch.cuda.current_device())
                pos_mask_sum = (minus_sin_weight != 1e+0).bool().int().sum(1) + sum1_mask.int()
                mean_log_prob_pos += ((minus_sin_weight != 1).bool().int() * log_prob ).sum(1) / pos_mask_sum

                # print(data_loader.label_emotion_dict)
                # print(labels)
                # print(minus_sin_weight[:2])
                # print(plus_pos_mask[:2])

                if bool(torch.isnan(mean_log_prob_pos).sum()):

                    print(log_prob[:3])
                    print(exp_logits[:3])
                    print(((mask * logits_mask).bool()))
                    print((log_prob * ((mask * logits_mask).bool().int())).sum(1))

                    exit()

            # loss
            loss = - mean_log_prob_pos
            loss = loss.view( batch_size).mean()
            return loss

#get f1 score
def f1_score(y_pred , y_true,weighted = True):
    epsilon = 1e-7
    class_num = len(y_pred[0])
    y_pred = torch.argmax(y_pred, dim=1)
    nums = torch.zeros((class_num,), device=y_pred.device)
    tp = torch.zeros((class_num,), device=y_pred.device)
    fp = torch.zeros((class_num,), device=y_pred.device)
    fn = torch.zeros((class_num,), device=y_pred.device)

    for i in range(class_num):
        nums[i] = (y_true==i).sum()
        tp[i] = torch.sum((y_true == i) & (y_pred == i))
        fp[i] = torch.sum((y_true != i) & (y_pred == i))
        fn[i] = torch.sum((y_true == i) & (y_pred != i))

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    if weighted:
        weighted_f1 = 0
        for i in range(class_num):
            weighted_f1 += f1[i] * nums[i] / len(y_true)
        return weighted_f1
    else:
        return torch.mean(f1)

#calculate accuracy for each class
def acc_each_class(y_pred , y_true):
    epsilon = 1e-7
    class_num = len(y_pred[0])
    y_pred = torch.argmax(y_pred, dim=1)
    tp = torch.zeros((class_num,), device=y_pred.device)
    fp = torch.zeros((class_num,), device=y_pred.device)
    fn = torch.zeros((class_num,), device=y_pred.device)

    for i in range(class_num):
        tp[i] = torch.sum((y_true == i) & (y_pred == i))
        fp[i] = torch.sum((y_true != i) & (y_pred == i))
        fn[i] = torch.sum((y_true == i) & (y_pred != i))

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)

    for i in range(class_num):
        emotion = data_loader.label_emotion_dict[i]
        print(f'{emotion} , truth num : {(y_true==i).sum()}')
        tp[i] = torch.sum((y_true == i) & (y_pred == i))
        fp[i] = torch.sum((y_true != i) & (y_pred == i))
        fn[i] = torch.sum((y_true == i) & (y_pred != i))
        print(f'{emotion} accuracy : {"%.2f" %(tp[i] / (tp[i] + fn[i]) * 100)}')
    print('')

