import matplotlib.pyplot as plt
import numpy as np

data_path = "./results/nodate_turb1_FEDformer_custom_withoudate_ftS_sl288_ll144_pl288_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0"


true = np.load(data_path+"/true.npy")
predict = np.load(data_path+"/pred.npy")

print(true.shape)
true = true[::16].reshape(-1)
predict = predict[::16].reshape(-1)
print(true.shape)

plt.plot(true,color='blue')
plt.plot(predict,color='red')
plt.show()
