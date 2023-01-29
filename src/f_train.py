import datetime
import os
import torch
import torch.optim.lr_scheduler as lr_scheduler
from f_dataloader import Data_Q_T,Data_Query
import parameters as par
from f_model import LSTM_ManyToOne
from tqdm import tqdm
import numpy as np
import h5py
torch.manual_seed(100)
from f_utils import calc_accuracy,eval_help,eval_variable_help,get_variable_legnth



class Trainer():
    def __init__(self,gpu,data_loader_train,data_loader_test,gallery_feat,test_labels,query_labels,model,loss,num_epochs=1,lr=0.001) -> None:
        self.model=model
        #self.batch_size=batch_size
        self.gallery_feat=gallery_feat
        self.test_labels =test_labels
        self.query_labels=query_labels
        #self.gt_label=
        self.gpu=gpu
        self.data_loader_train=data_loader_train
        self.data_loader_test=data_loader_test
        self.loss=loss
        self.lr=lr
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.lr,betas=(0.9, 0.999))
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer,par.step_decay, par.weight_decay)
        self.num_epochs=num_epochs
        self.date=datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.log_dir=os.path.join(par.LOG_DIR,self.date)
        os.makedirs(self.log_dir)
        if  torch.cuda.is_available():
            print("set cuda device")
            self.model.cuda()
        else:
            print('Warning: Using CPU')
    def train_variable(self):
        avg_loss = 0
        #self.model.zero_grad()
        self.model.train() #set the mode to train so it can update gradients
        tq=tqdm(self.data_loader_train)
        for i, sample in enumerate(tq):
            qFeat, tFeat, mani_vects,legnths = sample
            self.model.zero_grad()
            list_manips,id_x=get_variable_legnth(mani_vects)
            index_per=np.random.permutation(len(list_manips))
            out_batch=[]
            tFeat_batch=[]
            for n in index_per:
                if(len(list_manips[n])>0):
                    list_manips_n=torch.tensor(list_manips[n])
                    qFeat_n=qFeat[id_x[n]]
                    out,hidden = self.model(list_manips_n,qFeat_n)
                    tFeat_n=tFeat[id_x[n]]
                    tFeat_batch.append(tFeat_n)
                    out_batch.append(out)
            out_batch=torch.cat(out_batch, axis=0).cuda()
            tFeat_batch=torch.cat(tFeat_batch,axis=0).cuda()
            
            loss=self.loss(out_batch,tFeat_batch)
            loss.backward()
            self.optimizer.step()
            avg_loss+= loss.item()
                   
            tq.set_description("train: Loss batch numero :{ind} , value loss: {l}".format(ind=i, l=loss.item()))
        return avg_loss  / (i+1)
    def train(self):
        avg_loss = 0
        
        self.model.train() #set the mode to train so it can update gradients
        tq=tqdm(self.data_loader_train)
        for i, sample in enumerate(tq):
            qFeat, tFeat, mani_vects,legnths = sample
            self.model.zero_grad()
            out,hidden = self.model(mani_vects,qFeat)
            tFeat=tFeat.cuda()
            out.cuda()
            loss=self.loss(out,tFeat)#(batch_size,output_dim)
            tq.set_description("train: Loss batch numero :{ind} , value loss: {l}".format(ind=i, l=loss.item()))
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
        return avg_loss / (i+1)
   

    def eval(self):
        if(par.Eval_variable_legnth):
            predicted_tfeat=eval_variable_help(self.model,self.data_loader_test)
        else:
            predicted_tfeat =eval_help(self.model,self.data_loader_test)
        acc,res=calc_accuracy(self.gallery_feat,predicted_tfeat,self.query_labels,self.test_labels,par.K,par.N,dim = 4080)
        return acc
    
    def run(self):
        previous_best_avg_test_acc = 0.0
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            f.write("parameter of model:\nDataset:{data} N:{n},num of layer:{layer},CREATE_ZERO_MANIP_ONLY :{crea},MOVE_ZERO_MANIP_LAST:{move_zer},VAL_ORIGINAL:{val_orig},MODEL_EVAL:{model_eval}, Eval_variable_legnth:{eval_varia},Train_variable_legnth:{train_var},\n num_epoch:{epoch} ,lr:{lr},step_decay:{s},weight_decay:{dec},cont_training:{cont},pretrainde_model:{pretraind},".format(pretraind= par.pretrain_model,cont=par.contin_training,layer=par.NUM_LAYER,
                     n=par.N,crea=par.CREATE_ZERO_MANIP_ONLY,data=par.name_data_set ,move_zer=par.MOVE_ZERO_MANIP_LAST,
                     epoch=par.NUM_EPOCH,lr=par.LR,
                     val_orig=par.VAL_ORIGINAL,model_eval=par.MODEL_EVAL,
                     eval_varia=par.Eval_variable_legnth,train_var=par.Train_variable_legnth,
                     s=par.step_decay,dec=par.weight_decay))
        for epoch in range(self.num_epochs):
            if (par.Train_variable_legnth):
                avg_train_loss = self.train_variable()
            else:
                avg_train_loss=self.train()
            if (epoch%5==0):
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "ckpt_%d.pkl" % (epoch + 1)))
                print('Saved new checkpoint  ckpt_{epoch}.pkl , avr loss {l}'.format( epoch=epoch+1, l=avg_train_loss))
            avg_test_acc= self.eval()
            result="Epoch {e}, Train_loss: {l}, test_acc:{a} ,lr:{lr}\n".format(lr= self.lr_scheduler.get_lr(),e=epoch + 1,l=avg_train_loss,a=avg_test_acc)
            with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
                f.write(result)
            print(result)

        
            if avg_test_acc > previous_best_avg_test_acc:
                with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
                    f.write("saved_a new best_model\n ")
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "best_model.pkl"))
                print('Best model in {dir}/best_model.pkl'.format(dir=self.log_dir))
                previous_best_avg_test_acc = avg_test_acc

            self.lr_scheduler.step()


if __name__=="__main__":
   
    torch.cuda.set_device(1)
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('Loading dataset...')
    gallery_feat=np.load(par.FEAT_TEST_SENZA_N)
    test_labels = np.loadtxt(os.path.join(par.ROOT_DIR,par.LABEL_TEST), dtype=int)
    Data_test= h5py.File(par.DATA_TEST)
    t_id=Data_test['t']#id del target 
    query_labels=test_labels[t_id]
    #test_data =Data_Q_T(par.DATA_TEST,par.FEAT_TEST_SENZA_N,par.LABEL_TEST)

    test_data=Data_Query(Data_test=Data_test,gallery_feat=gallery_feat,label_data=test_labels,N=par.N)
    train_data =Data_Q_T(par.DATA_TRAIN,par.FEAT_TRAIN_SENZA_N,par.LABEL_TRAIN)
    
    
    train_loader=torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True,
                                               drop_last=True)
    test_loader=torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,
                                               sampler=torch.utils.data.SequentialSampler(test_data),
                                               drop_last=False)
    model=LSTM_ManyToOne(input_size=151,seq_len=par.N,output_size=4080,hidden_dim=4080,n_layers=par.NUM_LAYER,drop_prob=0.5)
    # create the folder to save log, checkpoints and args config
    if(par.contin_training):
        path_pretrained_model=os.path.join(par.LOG_DIR,"{pretrain_model}/best_model.pkl".format(pretrain_model=par.pretrain_model))
        model.load_state_dict(torch.load(path_pretrained_model))
        print("Pre trained model is loaded...")
    loss=torch.nn.MSELoss().cuda()
    trainer=Trainer(gpu=1,data_loader_train=train_loader, data_loader_test=test_loader,
    loss=loss,model=model,gallery_feat=gallery_feat,test_labels=test_labels,query_labels=query_labels,num_epochs=par.NUM_EPOCH,lr=par.LR)
    trainer.run()
    


