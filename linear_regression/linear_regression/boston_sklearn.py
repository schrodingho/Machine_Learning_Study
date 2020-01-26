from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

import numpy as np
loaded_data=datasets.load_boston()
data_X=loaded_data.data
data_y=loaded_data.target

X_train,X_test,y_train,y_test=train_test_split(data_X,data_y,test_size=0.2)#训练和测试集划分
model=LinearRegression()
model.fit(X_train,y_train)
print(model.coef_)
print(model.intercept_)

y_pred=model.predict(X_test)

print("MSE is:",metrics.mean_squared_error(y_test,y_pred))#LOSS=MSE均方差

#10折交叉验证
predicted=cross_val_predict(model,data_X,data_y,cv=10)
print("MSE is(with_cvp):",metrics.mean_squared_error(data_y,predicted))
plt.scatter(data_y,predicted,color='y',marker='o')
plt.scatter(data_y,data_y,color='g',marker='+')
plt.title('boston_house_price_prediction')
plt.xlabel('real_price')
plt.ylabel('predict_price')
plt.show()

