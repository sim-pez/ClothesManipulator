import datetime
import os
import torch
import torch.optim.lr_scheduler as lr_scheduler
from f_dataloader import Data_Q_T,fast_loader
import parameters as par
from f_model import LSTM_ManyToOne
from tqdm import tqdm

torch.manual_seed(100)


class Trainer():
    def __init__(self,gpu,data_loader_train,data_loader_test,model,loss,num_epochs=1,lr=0.001) -> None:
        self.model=model
        #self.batch_size=batch_size
        self.gpu=gpu
        self.data_loader_train=data_loader_train
        self.data_loader_test=data_loader_test
        self.loss=loss
        self.lr=lr
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.lr,betas=(0.9, 0.999))
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, 5, 0.1)
        self.num_epochs=num_epochs
        self.date=datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.log_dir=os.path.join(par.LOG_DIR,self.date)
        os.makedirs(self.log_dir)
    
        if  torch.cuda.is_available():
            print("sett cuda device")
            self.model.cuda()
        else:
            print('Warning: Using CPU')
            
    def train(self):
        avg_loss = 0
        #self.model.zero_grad()
        self.model.train() #set the mode to train so it can update gradients
        tq=tqdm(self.data_loader_train)
        for i, sample in enumerate(tq):
            qFeat, tFeat, mani_vects,id_t = sample
            
            mani_vects.cuda()
            self.model.zero_grad()
            out,hidden = self.model(mani_vects,qFeat)
            out.cuda()
            loss=self.loss(out,tFeat)#(batch_size,output_dim)
            tq.set_description("train: Loss batch numero :{ind} , value loss: {l}".format(ind=i, l=loss.item()))
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
        return avg_loss / (i+1)
        
    def eval(self):
        self.model.eval()
        avg_loss=0
        with torch.no_grad():
            tq=tqdm(self.data_loader_test)
            for i, sample in enumerate(tq):
                qFeat,tFeat,mani_vects,id_t = sample
                
                
                out,hidden = self.model(mani_vects,qFeat)
                tFeat.cuda()
                out.cuda()
                loss=self.loss(out,tFeat)#(batch_size,output_dim   
                tq.set_description("eval:Loss batch numero :{ind} , value loss: {l}".format(ind=i, l=loss.item()))
                         
                avg_loss += loss.item()
                
        return avg_loss/(i+1)

    def run(self):
        previous_best_avg_test_loss = 1000000
        for epoch in range(self.num_epochs):
            avg_train_loss = self.train()
            torch.save(self.model.state_dict(), os.path.join(self.log_dir, "ckpt_%d.pkl" % (epoch + 1)))
            print('Saved checkpoints at {dir}/ckpt_{epoch}.pkl'.format(dir=self.log_dir, epoch=epoch+1))
            
            avg_test_loss = self.eval()

            result="Epoch {e}, Train_loss: {l}, test_loss:{a} \n".format(e=epoch + 1,l=avg_train_loss,a=avg_test_loss)
            with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
                f.write(result)
            print(result)

            # store parameters
            

            if avg_test_loss < previous_best_avg_test_loss:
                with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
                    f.write("saved_a new best_model\n ")
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "best_model.pkl"))
                print('Best model in {dir}/best_model.pkl'.format(dir=self.log_dir))
                previous_best_avg_test_loss = avg_test_loss

            self.lr_scheduler.step()


if __name__=="__main__":
   # print(torch.cuda.memory_stats())
    torch.cuda.set_device(1)
    #torch.multiprocessing.set_start_method('spawn')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # load dataset
    print('Loading dataset...')
    train_data =Data_Q_T(par.DATA_TRAIN,par.FEAT_TRAIN_SENZA_N,par.LABEL_TRAIN,shuffle=True)
    test_data =Data_Q_T(par.DATA_TEST,par.FEAT_TEST_SENZA_N,par.LABEL_TEST,shuffle=False)
    
    train_loader=torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True,
                                               
                                               drop_last=True)
    test_loader=torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False,
                                               sampler=torch.utils.data.SequentialSampler(test_data),
                                               
                                               drop_last=False)
    model=LSTM_ManyToOne(input_size=151,seq_len=8,output_size=4080,hidden_dim=4080,n_layers=par.NUM_LAYER,drop_prob=0.5)
    # create the folder to save log, checkpoints and args config
    
    loss=torch.nn.MSELoss().cuda()
    trainer=Trainer(gpu=1,data_loader_train=train_loader, data_loader_test=test_loader,
    loss=loss,model=model,num_epochs=10,lr=0.001)
    trainer.run()
    


