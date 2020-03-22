import sys
import pandas as pd
import numpy as np
import random

inputPath = sys.argv[1]
outputPath= sys.argv[2]
print('inp',inputPath)
w = np.load('weight.npy')
GoodFeature = np.load('GoodFeature.npy')
GoodNum = np.sum(GoodFeature)
mean_x = np.load('mean_x.npy')
std_x = np.load('std_x.npy')

#test 
testdata = pd.read_csv(inputPath, header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, GoodNum*9], dtype = float)
for i in range(240):
    #test_data[18 * i+14,:],test_data[18 * i+17,:] = P2C(test_data[18 * i+17,:],test_data[18 * i+14,:])
    #test_data[18 * i+15,:],test_data[18 * i+16,:] = P2C(test_data[18 * i+16,:],test_data[18 * i+15,:])
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :][GoodFeature,:].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)



print('w=',max(w),min(w))
ans_y = np.dot(test_x, w)

import csv
with open(outputPath, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        #print(row)