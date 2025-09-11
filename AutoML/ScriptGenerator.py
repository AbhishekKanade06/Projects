import pandas as pd

class ScriptGenerator:

    def __init__(self, info,target):
        self.info = info
        self.target=target
        self.script=''
    
    def create_script(self):
        self.script+=dependancy
        self.script+=f"info={self.info} \ntarget='{self.target}'"
        self.script+=load_script
        self.script+=input_script
        self.script+=prediction_script
        # self.Gen_script()

dependancy ="""

import pandas as pd
import pickle
import os

"""

load_script="""

path=os.getcwd()

model_path=path+"/tuned_model.pkl"
lable_endocer=path+'/label.pkl'
scaler = path+'/scaler.pkl'
# Load model,encoder,scalar
model = pickle.load(open(model_path, 'rb'))
encoder = pickle.load(open(lable_endocer, 'rb'))
scaler = pickle.load(open(scaler, 'rb'))

"""



input_script="""

data=[]
for col in info.keys():
    if col == target:
        continue
    if info[col] == 'object':
        while True:
            x=input(f"Enter {col} value from {encoder[col].classes_}: ")
            try:
                encoded_val = encoder[col].transform([x])[0]
                scaled_val = scaler[col].transform(pd.DataFrame({col: [encoded_val]}))[0][0]
                data.append(scaled_val)
                break
            except:
                print(f"Invalid value for {col}. Please try again.")
                 
    if info[col] == 'int64' or info[col] == 'float64':
        while True:
            try:
                x=float(input(f"Enter {col} value: "))
                scaled_val = scaler[col].transform(pd.DataFrame({col: [x]}))[0][0]
                data.append(scaled_val) 
                break
            except:
                print(f"Invalid value for {col}. Please try again.")

"""

prediction_script="""

if info[target] == 'object':
    print(f"Predicted {target} is :{encoder[target].inverse_transform([model.predict([data])])[0]}")  
else:
    print(f"Predicted {target} is : {model.predict([data])[0]}")      

"""
  