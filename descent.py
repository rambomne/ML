import numpy as np
import pandas as pd
d=pd.read_html("https://github.com/akiwelekar/MLModels/blob/master/aimarks2017.csv")
df=d[0]
m=0

c=0

lr=0.00001

epoch=10000

mse_mark =np.array(df["mse"])
ese_mark = np.array(df["ese"])

n=float(len(mse_mark))

for i in range(epoch):
    y_pred = mse_mark * m + c
    dm = (-2/n) * sum ( mse_mark * (ese_mark - y_pred))
    de = (-2/n) * sum (ese_mark - y_pred)
    m = m - lr * dm
    c = c - lr * de
print("%.3f"%(m))
print("%.3f"%(c))
