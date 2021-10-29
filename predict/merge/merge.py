import numpy as np
import time

import pandas as pd

result = pd.read_csv('../result/CSTModel-09212209-e(100)-b(64)-eta(1e-06)-loss(480.9583).h5.csv')

submit = pd.read_csv('PreliminaryValidationSet_Info.csv')

continue_injection = '1028/continue_injection_pred.txt'
HRF = '1028/HRF_pred.txt'
IRF = '1028/IRF_pred.txt'
PED = '1028/PED_pred.txt'
SRF = '1028/SRF_pred.txt'
VA = '1028/VA_pred.csv'

# data = np.loadtxt(continue_injection)
# data = pd.Series(data, name='continue injection', dtype=np.int)
# submit = pd.concat([submit, data], axis=1)

# data = np.loadtxt(HRF)
# data = pd.Series(data, name='HRF', dtype=np.int)
# submit = pd.concat([submit, data], axis=1)

# data = np.loadtxt(IRF)
# data = pd.Series(data, name='IRF', dtype=np.int)
# submit = pd.concat([submit, data], axis=1)

# data = np.loadtxt(PED)
# data = pd.Series(data, name='PED', dtype=np.int)
# submit = pd.concat([submit, data], axis=1)

# data = np.loadtxt(SRF)
# data = pd.Series(data, name='SRF', dtype=np.int)
# submit = pd.concat([submit, data], axis=1)

# data = pd.read_csv(VA, index_col=0)
# data = pd.Series(np.around(data.iloc[:, 0], decimals=3), name='VA', dtype=np.float)
# submit = pd.concat([submit, data], axis=1)
# submit = submit[['patient ID', 'continue injection', 'VA']]

data = pd.read_csv('1029/new_data_pred.csv', index_col=0)
submit = pd.concat([submit, data], axis=1)

data = pd.read_csv('1029/VA_pred.csv', index_col=0)

data1 = data[['continue injection', 'VA_1']].rename(columns={'VA_1': 'VA'})
data2 = data[['continue injection', 'VA_2']].rename(columns={'VA_2': 'VA'})
data3 = data[['continue injection', 'VA_3']].rename(columns={'VA_3': 'VA'})
data4 = data[['continue injection', 'VA']]

submit = pd.concat([submit, data4], axis=1)

submit = submit[['patient ID', 'continue injection', 'IRF', 'SRF', 'HRF', 'VA']]
result = result[['patient ID', 'preCST', 'CST']]
result = result.round(3)
result = pd.merge(result, submit, on='patient ID')
result.loc[result['preCST'].isna(), 'preCST'] = result['preCST'].mean()
result.loc[result['CST'].isna(), 'CST'] = result['CST'].mean()
result = result[['patient ID', 'preCST', 'VA', 'continue injection', 'CST', 'IRF', 'SRF', 'HRF']]
result.to_csv(f'result-{time.strftime("%m%d%H%M")}.csv', index=False)

print(result.head())
