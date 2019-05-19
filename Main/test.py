

from sklearn.metrics import confusion_matrix


tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1,1,1], [1,1, 1, 1, 0,1]).ravel()


[TP, FP], [FN, TN] = confusion_matrix([0, 1, 0, 1,1,1], [1,1, 1, 1, 0,1]) #.ravel()



import numpy as np
a,b,c,d = np.load("/Users/dominiquepaul/xCoding/classification_tool/Data/np_files_final/food_final_testing_dataset.npy")

sum(b)/len(b)


confusion_matrix([0, 1, 0, 1,1,1], [1,1, 1, 1, 0,1]) #.ravel()
