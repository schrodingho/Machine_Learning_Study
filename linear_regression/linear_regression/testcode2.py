import pandas as pd
import numpy as np
def dataprocess(df):
    xlist=[]
    ylist=[]
    df=df.replace(['NR'],[0.0])
    array=np.array(df).astype(float)

    for i in range(0,4320,18):
        for j in range(24-9):
            mat=array[i:i+18,j:j+9]
            label=array[i+9,j+9]
            xlist.append(mat)
            ylist.append(label)
    x=np.array(xlist)
    y=np.array(ylist)

    return x,y,array

def train(x_train,y_train,epoch):
    bias=0
    weights=np.ones(9)
    learning_rate=1
    reg_rate=0.001
    bg2_sum=0
    wg2_sum=np.zeros(9)

    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(9)
        for j in range(3200):
            b_g += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) * (-1)
            for k in range(9):
                w_g[k] += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) * (-x_train[j, 9, k])
        b_g/=3200
        w_g/=3200

        for m in range(9):
            w_g[m]+=reg_rate*weights[m]

        #adagrad
        bg2_sum=b_g**2
        wg2_sum+=w_g**2

        bias-=learning_rate/bg2_sum**0.5*b_g
        weights-=learning_rate/wg2_sum**0.5*w_g

        if i%200==0:
            loss=0
            for j in range(3200):
                loss+=(y_train[j]-weights.dot(x_train[j,9,:])-bias)**2
            print(f"after {i} epochs, the loss on train data is {loss/3200}.")

    return weights,bias

def validation(x_val,y_val,weights,bias):
    loss=0
    for i in range(400):
        loss+=(y_val[i]-weights.dot(x_val[i,9,:])-bias)**2
    return loss/400


if __name__=='__main__':
    df=pd.read_csv('pm2.5.csv',usecols=range(3,27))
    x,y,array=dataprocess(df)
    x_train,x_validation=x[0:3200],x[3200:3600]
    y_train,y_validation=y[0:3200],y[3200:3600]

    epoch=2000
    w,b=train(x_train,y_train,epoch)
    val_loss=validation(x_validation,y_validation,w,b)
    print(f"the loss on validation dataset is {val_loss}")

