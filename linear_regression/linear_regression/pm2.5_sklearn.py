from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def dataprocess(df):
    xlist=[]
    ylist=[]
    df=df.replace(['NR'],[0.0])
    array=np.array(df).astype(float)

    for i in range(0,4320,18):
        for j in range(24-9):
            mat=array[i:i+18,j:j+9]
            label=array[i+9,j+9]
            xlist.append(mat[9,:])
            ylist.append(label)
    x=np.array(xlist)
    y=np.array(ylist)

    return x, y

df = pd.read_csv('pm2.5.csv', usecols=range(3, 27))
x, y = dataprocess(df)


# loaded_data=datasets.load_boston()
# data_X=loaded_data.data
# data_y=loaded_data.target

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)#训练和测试集划分
model=LinearRegression()
model.fit(X_train,y_train)
print(model.coef_)
print(model.intercept_)

y_pred=model.predict(X_test)

print("MSE is:",metrics.mean_squared_error(y_test,y_pred))#LOSS=MSE均方差

#10折交叉验证
predicted=cross_val_predict(model,x,y,cv=10)
print("MSE is(with_cvp):",metrics.mean_squared_error(y,predicted))
plt.scatter(y,predicted,color='y',marker='o')
plt.scatter(y,y,color='g',marker='+')
plt.title('pm2.5_prediction')
plt.xlabel('real_pm2.5')
plt.ylabel('predict_pm2.5')
plt.show()

