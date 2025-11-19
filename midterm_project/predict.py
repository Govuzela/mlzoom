
import pickle
import pandas as pd
### Parameters
depth= 10
learning_rate=0.02
iterations =200

params_str = 'depth_%g_learning_rate_%.3f_iterations_%g'%(depth,learning_rate,iterations)
mine_code ={1.0: 'No_mine',2.0: 'Anti_tank',3.0: 'Anti_personnel',4.0: 'Booby_trapped/Anti_personnel',5.0: 'M14_anti_personnel'}
# ### Load model

outfile ='model_depth_10_lr_0.02_iter_200.bin'


with open(outfile,'rb') as f_in:
    dv,model = pickle.load(f_in)
    dv,model
import pandas as pd

# Single mine input
import pandas as pd

# Example single input
mine = {'Voltage': 0.240966462513951,
        'Height': 0.727272727272727,
        'soil_type_cat': 1,
        'Mine_type': 1.0}  # True label

# Transform using DictVectorizer
X_vec = dv.transform([mine])

# Predict probabilities
y_pred_probs = model.predict_proba(X_vec)[0]  # array of class probabilities

# Predict class
prediction = model.predict(X_vec)[0]  # scalar

# Map predicted class to human-readable label
predicted_label = mine_code[float(prediction)]

# Map true target to human-readable label
true_label = mine_code[float(mine['Mine_type'])]

print('Input features (without target):', {k: v for k, v in mine.items() if k != 'Mine_type'})
print('True class label:', true_label)
print('Predicted probability of each class:', y_pred_probs)
print('Predicted class label:', predicted_label)
