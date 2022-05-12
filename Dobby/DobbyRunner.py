import sys

sys.path.insert(0, '/home/williams/github/Quantium/')

from Dobby.DobbyTrainer import DobbyTrainer
from ModelBuilder.cnn import cnn
from ModelBuilder.resnet import resnet
from ModelBuilder.residual import residual
from ModelBuilder.mobilenet import mobilenet

print('running mobilenet')
model = DobbyTrainer(json='/home/williams/github/Quantium/dataset2-flip/json/train.json', kp_def='/home/williams/github/Quantium/dataset2-flip/dataset_head.csv',
                     images='/home/williams/github/Quantium/dataset2-flip/img/train/', checkpoint_dir='checkpoints/', log_dir='logs/',save_dir='models/', name='McFly_mobilenet', model=mobilenet())
model.train()

print('running resnet')
model = DobbyTrainer(json='/home/williams/github/Quantium/dataset2-flip/json/train.json', kp_def='/home/williams/github/Quantium/dataset2-flip/dataset_head.csv',
                     images='/home/williams/github/Quantium/dataset2-flip/img/train/', checkpoint_dir='checkpoints/', log_dir='logs/',save_dir='models/', name='McFly_mobilenet', model=resnet())
model.train()

print('running residual')
model = DobbyTrainer(json='/home/williams/github/Quantium/dataset2-flip/json/train.json', kp_def='/home/williams/github/Quantium/dataset2-flip/dataset_head.csv',
                     images='/home/williams/github/Quantium/dataset2-flip/img/train/', checkpoint_dir='checkpoints/', log_dir='logs/',save_dir='models/', name='McFly_mobilenet', model=residual())
model.train()

print('running cnn')
model = DobbyTrainer(json='/home/williams/github/Quantium/dataset2-flip/json/train.json', kp_def='/home/williams/github/Quantium/dataset2-flip/dataset_head.csv',
                     images='/home/williams/github/Quantium/dataset2-flip/img/train/', checkpoint_dir='checkpoints/', log_dir='logs/',save_dir='models/', name='McFly_mobilenet', model=cnn())
model.train()