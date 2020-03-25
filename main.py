import sys
import pandas as pd
import numpy as np
import random

data = pd.read_csv('./train.csv', encoding = 'big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()
'''
def P2C(r,the):
    r = r.astype(np.float)
    the = the.astype(np.float)
    return r*np.cos(np.deg2rad(the)), r*np.sin(np.deg2rad(the))'''

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
        #sample[14,:],sample[17,:] = P2C(sample[17,:],sample[14,:])
        #sample[15,:],sample[16,:] = P2C(sample[16,:],sample[15,:])
    month_data[month] = sample


y = np.empty([12 * 471, 1], dtype = float)
corr = np.empty([18,12*471],dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            #x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 5].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
            corr[:,month * 471 + day * 24 + hour]=month_data[month][:, day * 24 + hour]

print(np.corrcoef(corr).shape)
corr = np.corrcoef(corr)[9]
GoodFeature = abs(corr)>0.2
GoodNum = np.sum(GoodFeature)
#print(corr,GoodFeature,GoodNum)
x = np.empty([12 * 471, GoodNum * 9], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][GoodFeature,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
print(x.shape)

#normalize
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]




dim = GoodNum * 9 + 1

x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)#, x**2
w = np.zeros([dim,1])
#np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(),x)),x.transpose()),y)
print('x shape',x.shape)
learning_rate = 50#10*10**(best_case%3)
iter_time = 10000#10*10**(case%9//3)
landa = 10#10*10**(case//9)
adagrad = np.zeros([dim, 1])
mt = np.zeros([dim, 1])
vt = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    if(t%100==0):
        print(str(t) + ":" + str(loss))
    #z = np.dot(x, w) #(471*12)*1
    #yhat = 1/(1+np.exp(-z))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w)-y)+2*landa*w #dim*1
    adagrad += gradient ** 2
    #mt = beta1*mt + (1-beta1)*gradient
    #vt = beta2*vt + (1-beta2)*gradient**2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    #w = w - learning_rate * mt/(1-np.power(beta1,t+1)) / (np.sqrt(vt/(1-np.power(beta2,t+1)))+eps)
np.save('weight.npy', w)
np.save('GoodFeature.npy',GoodFeature)
np.save('mean_x.npy',mean_x)
np.save('std_x.npy',std_x)

