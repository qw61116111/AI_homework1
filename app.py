import argparse
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
import csv    




num_day=7

data_fold=['MW']
label_fold=['MW']
'''
train_mean=[33539.8,29950.6,3498.3,261.0]
train_std=[4309.5, 3693.5,1300.9, 43.6]
'''


train_mean=[3484.3]

train_std=[1168.53]


num_train=810
class dataset(torch.utils.data.Dataset):

    def __init__(self,is_train=True):
        self.data=[]
        self.label=[]
        self.temp=[]
 
        
        for i in range(len(data_csv[:num_train])):
            for j in range(len(data_fold)):
                self.temp.append(data_csv[data_fold[j]][i])
            self.data.append(self.temp)
            self.temp=[]
            
        z=np.array(self.data).T
        train_mean=np.mean(z,axis=1)
        train_std=np.std(z,axis=1)

        z=[]
        self.data=[]
        if is_train:
            '''--------------train_set---------------'''
            for i in range(len(data_csv[:num_train])):
                for j in range(len(data_fold)):
                    self.temp.append(data_csv[data_fold[j]][i])

                self.data.append(self.temp)
                self.temp=[]
            
            z=np.array(self.data).T
            '''
            train_mean=np.mean(z,axis=1)
            train_std=np.std(z,axis=1)
            '''
            for i in range(len(data_fold)):
                for j in range(len(data_csv[:num_train])):

                    z[i][j]-=train_mean[i]
                    z[i][j]/=train_std[i]
            self.data=[]
            self.data=z.T
        
            for i in range(len(data_csv[:num_train])):
                self.label.append(data_csv[label_fold[0]][i])
                
        else:
            '''-----------------val_set----------------'''  
            self.data=[]
            for i in range(len(data_csv[num_train:])):
                for j in range(len(data_fold)):
                    self.temp.append(data_csv[data_fold[j]][num_train+i])

                self.data.append(self.temp)
                self.temp=[]
            
            z=np.array(self.data).T
            for i in range(len(data_fold)):
                for j in range(len(data_csv[num_train:])):

                    z[i][j]-=train_mean[i]
                    z[i][j]/=train_std[i]
            self.data=[]
            self.data=z.T
        
            for i in range(len(data_csv[num_train:])):
                #print(self.num_train+i,data_csv[label_fold[0]][self.num_train+i])
                self.label.append(data_csv[label_fold[0]][num_train+i])
                
    def __len__(self):

        return len(self.data)-num_day
    
    def __getitem__(self, index):

        #print(index)

        a=(self.label[index+num_day])
        #print(a)

        return self.data[index:index+num_day],a
#%%
'''
train_mean=[33539.8,29950.6,3498.3,261.0]
train_std=[4309.5, 3693.5,1300.9, 43.6]
'''
'''
train_mean=[3498.3]
train_std=[1300.9]
'''

train_mean=[3484.3]

train_std=[1168.53]

def test_pred(train_mean,train_std):
    data=[]
    label=[]
    temp=[]

    for i in range(len(data_csv[num_train-num_day:num_train])):
        for j in range(len(data_fold)):

            temp.append(data_csv[data_fold[j]][num_train-num_day+i])
        data.append(temp)
        temp=[]

    z=np.array(data).T


    for i in range(len(data_fold)):
        for j in range(len(data_csv[num_train-num_day:num_train])):

            z[i][j]-=train_mean[i]
            z[i][j]/=train_std[i]
    data=[]
    data=z.T
    data=data[np.newaxis,:]

    return torch.from_numpy(data)

            


#%%
epochs = 1500

batch_size = 64

hidden_size = 32
num_layers = 2
num_feature=len(data_fold)



class LSTM(nn.Module):
    def __init__(self,num_feature,hidden_size, num_layers):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size=num_feature,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
            )
        

        self.fc=nn.Linear(hidden_size,1)
        self.fc1=nn.Linear(num_day,1)
        self.ReLU=nn.PReLU()
        self.dropout = nn.Dropout(p=0.5)



    def forward(self,inputs):

        out,(h_n,c_n)=self.lstm(inputs, None)

        outputs=self.fc(out)
        
        outputs=torch.squeeze( outputs,2)

        outputs=self.fc1(outputs)

        return  outputs


def MSE(y_pred,y_true): 
    return  torch.mean((y_pred-y_true)**2)

#%%
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()
    data_csv = pd.read_csv(args.training)
    #%%
    net=LSTM(num_feature ,hidden_size,num_layers)
    trainloader=DataLoader(dataset(is_train=True),batch_size=batch_size,shuffle=False)
    net.cuda()
    #%%
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss(reduction='mean')
    z=0
    date=20210322
    pred=[]
    for i in range(epochs):
        z=0
        for num_batch,data in enumerate(trainloader,0):
            net.train()
            inputs,label=data
            inputs,label=inputs.float().cuda(),label.float().cuda()
            out=net(inputs)
    
            loss=MSE(torch.squeeze(out),label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            z+=loss.item()
        print('train_loss= %.2f,  %d epoch left'%(np.sqrt(z/num_batch),epochs-i))
        
        if i%100==0:
            z=0

            test_input=test_pred(train_mean,train_std)

            with torch.no_grad():
                pred=[]
                with open(args.output, 'w', newline='') as csvfile:
                    for i in range(8):
                        
                        net.eval()
                        
                        
                        test_input=test_input.float().cuda()
                        test_out=net(test_input)
                        pred.append(test_out)
    
                        if(i!=0 and 1):
                            writer = csv.writer(csvfile)
                            writer.writerow([date+i, test_out.item()])

                        for i in range(num_day):
                            if(i!=num_day-1):
                                test_input[0][i][0]=test_input[0][i+1][0]
                            else :
                                test_out=(test_out-3484.3)/1168.53
                                test_input[0][i][0]=test_out
#%%
        


            

    
#%%

    
