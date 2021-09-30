import pickle
import numpy as np

import keras

from sklearn.metrics import classification_report, confusion_matrix

# DataSet Paths
test_set_path = '/data/shared/sahan_dabarera/Image_Classification_Challange/Data/PKL_Data/Test_Data.pkl'

# Saved Models Path
nation_model_path = '/data/shared/sahan_dabarera/Image_Classification_Challange/Saved_Models/Inception_Nation/20210919-093406'
gender_model_path = '/data/shared/sahan_dabarera/Image_Classification_Challange/Saved_Models/Xception_Gender/20210918-140109'


nation_model = keras.models.load_model(nation_model_path)
gender_model = keras.models.load_model(gender_model_path)


test_data = pickle.load(open(test_set_path, 'rb'))

X_test = test_data['X_train']
Yn_test = test_data['Yn']
Yg_test = test_data['Yg']

pred_yn = nation_model.predict(X_test)
pred_yg = gender_model.predict(X_test)

pred_yn = (pred_yn > 0.5).astype('int')
pred_yg = (pred_yg > 0.5).astype('int')

# Calculate Combined Predictions
Y_test = Yn_test.astype('int')*2 + Yg_test.astype('int')
Y_pred = pred_yn*2 + pred_yg

# Confusion Matrix
print("Confusion Matrix")
print("Horz (True) | Vert (Pred) | Caucasian Woman, Caucasian Man, Indian Woman, Indian Man")
print(confusion_matrix(Y_test, Y_pred))

# Classification Report
print("Classification Report")
print(classification_report(Y_test, Y_pred, \
	target_names = ['Caucasian Woman(Class 0)', 'Caucasian Man(Class 1)', 'Indian Woman(Class 2)', 'Indian Man(Class 3)']))