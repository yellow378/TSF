import matplotlib.pyplot as plt
import numpy as np

data_path = "./results/nodate_turb1_speed_Informer_custom_withoudate_ftMS_sl288_ll144_pl288_dm2048_nh12_el5_dl2_df1024_expand2_dc4_fc1_ebtimeF_dtTrue_test_0"


true = np.load(data_path+"/true.npy")
predict = np.load(data_path+"/pred.npy")

print(true.shape)
true = true[::16].reshape(-1)
predict = predict[::16].reshape(-1)
print(true.shape)

plt.plot(true,color='blue')
plt.plot(predict,color='red')
plt.show()
