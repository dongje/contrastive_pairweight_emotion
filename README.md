# contrastive_pairweight_emotion

this is the implement about my first paper


the emotion should be uniformed in data and not label (order not relevant)
(neutral , sad , angry , surprise , happy , fear , disgust)

###### input example: 
python main.py --device 0 --train_data dataset/1dialog_kor/1dialog_kor_train.csv --valid_data dataset/1dialog_kor/1dialog_kor_valid.csv 
--max_length 64 --batch_size 128 --lang kr --n_epochs 200  --ce_alpha 1 --scl_lr 1e-5 --bert_freezing 1 
--pair_weight_exp 1.2 1.4 2.0  1.2 1.4 2.0  1.2 1.4 2.0  1.2 1.4 2.0 0.8 0.8 0.8 0.8

you can input pair_weight_exp array at once for efficient experiment

save best performance model automatically after training (including hyper parameter info all)


before start , be sure create forder named 'models' for saving!!
---
=======
emotion classify label should be uniformed. it's not number (order is not relevant)

neutral , happy , surprise , fear , disgust , sad , angry

can input many weights at once for efficient experiments

###### input example (negative): 
python main.py --device 0 --train_data dataset/1dialog_kor/1dialog_kor_train.csv --valid_data dataset/1dialog_kor/1dialog_kor_valid.csv --max_length 64 --batch_size 128 --lang kr --n_epochs 200 --ce_alpha 1 --scl_lr 1e-5 --bert_freezing 1 --pair_weight_exp 1.2 1.4 2.0 1.2 1.4 2.0 1.2 1.4 2.0 1.2 1.4 2.0 0.8 0.8 0.8 0.8 --only_best_check

###### input example (positive + curriculum):
python main.py --device 0 --train_data dataset/1dialog_kor/1dialog_kor_train.csv --valid_data dataset/1dialog_kor/1dialog_kor_valid.csv --max_length 64 --batch_size 128 --lang kr --n_epochs 200  --ce_alpha 1 --scl_lr 5e-6 --bert_freezing 1 --pair_weight_exp 2.0 1.4 2.0 1.4  --weight_where positive --decrease_curl_rate 0.1 --only_best_check


can input many weight_exp. for many experiment for once


be sure that the forder named models exists in directory!! (for saving model)
>>>>>>> 001e1ab95f858d82a8abf899a21ef21512ae6510
