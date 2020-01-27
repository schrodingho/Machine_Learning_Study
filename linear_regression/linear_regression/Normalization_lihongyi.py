import pandas as pd
import numpy as np
df = pd.read_csv('pm2.5.csv', usecols=range(3, 27))
df=df.replace(['NR'],[0.0])
array=np.array(df).astype(float)

#row行，column列
print(array[0])
for i in range(4320):
    mean=np.mean(array[i])
    std=np.std(array[i], ddof=1)
    for j in range(24):
        array[i][j]=(array[i][j]-mean)/std

print(array.shape,array[0])
