import numpy as np

npz_data = np.load('./data/ntu120/NTU120_CSet_ori.npz')
print(npz_data['x_train'].shape)