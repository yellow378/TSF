import matplotlib.pyplot as plt
import numpy as np

data_path = "./results/nodate_Trub1_wind_PatchTST_custom_withoudate_ftMS_sl400_ll360_pl36_dm512_nh9_el8_dl6_df512_expand2_dc4_fc1_ebtimeF_dtTrue_test_0"


true = np.load(data_path+"/true.npy")
predict = np.load(data_path+"/pred.npy")

print(true.shape)
true = true[::36].reshape(-1)
predict = predict[::36].reshape(-1)
print(true.shape)

plt.plot(true,color='blue')
plt.plot(predict,color='red')
plt.show()
