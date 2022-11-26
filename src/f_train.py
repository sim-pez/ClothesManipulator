
import argparse
import datetime
import json

import constants as C
import os

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from f_dataloader import Data_Q_T,fast_loader
import parameters as par

from f_model import LSTM_ManyToOne
from tqdm import tqdm

torch.manual_seed(100)


class Trainer():
    def __init__(self,batch_size,gpu,data_loader_train,data_loader_test,model,optimizer,loss,num_epochs=1) -> None:
        self.model=model
        self.batch_size=batch_size
        self.gpu=gpu
        self.data_loader_train=data_loader_train
        self.data_loader_test=data_loader_test
        self.loss=loss
        self.optimizer=optimizer
        self.lr_scheduler = lr_scheduler.StepLR(optimizer, 10, 0.5)
        self.num_epochs=num_epochs
        self.log_dir=datetime.datetime.now().strftime("%m-%d-%H:%M")
        os.makedirs(self.log_dir)
        self.cpu=False
        
        if  torch.cuda.is_available() and self.gpu!=None:
            torch.cuda.set_device(self.gpu)
            self.model.cuda()
        else:
            self.cpu=True
            print('Warning: Using CPU')
            
    def train(self):
        avg_loss = 0
        
        self.model.train()#set the mode to train so it can update gradients.
        for i, sample in enumerate(tqdm(train_loader)):

            qFeat,tFeat,mani_vects = sample
            if not self.cpu:
                qFeat = qFeat.cuda()
                tFeat=tFeat.cuda()
                mani_vects=mani_vects.cuda()
            self.model.zero_grad() 
            ####chaima la funzione forword in model
            #TODO hidden matrix of dim n_layers, batch,hidden_dim
            hidden =qFeat# TODO some operation
            out,hidden = self.model(mani_vects,hidden)
            loss=self.loss(out,tFeat)#(batch_size,output_dim)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss
        return avg_loss / (i+1)
        
    def eval(self):
        pass
    def run(self):
        previous_best_avg_test_acc = 0.0
        for epoch in range(self.num_epochs):
            avg_total_loss = self.train()
            avg_test_acc = self.eval()

            # result record
            result="Epoch {e}, Train_loss: {l}, test_acc:{a} \n".format(e=epoch + 1,l=avg_total_loss,a=avg_test_acc)
            with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
                f.write(result)
            print(result)

            # store parameters
            torch.save(self.model.state_dict(), os.path.join(self.log_dir, "ckpt_%d.pkl" % (epoch + 1)))
            print('Saved checkpoints at {dir}/ckpt_{epoch}.pkl'.format(dir=directory, epoch=epoch+1))

            if avg_test_acc > previous_best_avg_test_acc:
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "best_model.pkl"))
                print('Best model in {dir}/extractor_best.pkl'.format(dir=self.log_dir))
                previous_best_avg_test_acc = avg_test_acc

            self.lr_scheduler.step()


if __name__=="main":
    lr=0.005
  
    # load dataset
    print('Loading dataset...')
    train_data =Data_Q_T(par.DATA_TRAIN_DIR,par.DATA_TRAIN,mode1="train",shuffle=True)
    test_data =Data_Q_T(par.DATA_TRAIN_DIR,par.DATA_TRAIN,mode1="train",shuffle=False)
    
    train_loader=fast_loader(train_data,batch_size=300)
    test_loader=fast_loader(test_data,batch_size=300,shuffle=False,drop_last=False)
    model=LSTM_ManyToOne(input_size=151,seq_len=8,output_size=4080,hidden_dim=4080,n_layers=1)
    # create the folder to save log, checkpoints and args config
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    loss=torch.nn.MSELoss()
    trainer=Trainer(batch_size=300,)
    



