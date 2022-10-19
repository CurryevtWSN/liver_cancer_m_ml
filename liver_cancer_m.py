import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
#应用标题
st.set_page_config(page_title='Prediction model for metastasis of hepatocellular carcinoma')
st.title('Machine Learning for metastasis of primary liver cancer Identification : Development and Validation of a Prediction Model')
st.sidebar.markdown('## Variables')
Solitary = st.sidebar.selectbox('Solitary',('No','Yes'),index=0)
Vascular_invasion = st.sidebar.selectbox('Vascular_invasion',('No','Yes'),index=1)
AFP_400 = st.sidebar.selectbox('AFP400',('<400','≥400'),index=1)
Age = st.sidebar.slider("Age(year)", 0, 99, value=45, step=1)
PreAlb = st.sidebar.slider("Albumin", 0.0, 99.0, value=38.7, step=0.1)
PreGGT = st.sidebar.slider("Gamma-glutamyl transpeptidase", 0, 500, value=284, step=1)
PreALP = st.sidebar.slider("Alkaline phosphatase", 0, 400, value=81, step=1)
PreFib = st.sidebar.slider("Fibrinogen", 0.00, 20.00, value=5.05, step=0.01)
N = st.sidebar.slider("Neutrophil granulocyte", 0.00, 20.00, value=3.72, step=0.01)
Plt = st.sidebar.slider("Platelet", 0, 500, value=189, step=1)
NLR = st.sidebar.slider("The ratio of neutrophils to lymphocytes", 0.00, 30.00, value=2.93, step=0.01)
PLR = st.sidebar.slider("The ratio of platelet to lymphocytes", 0.00, 500.00, value=148.82, step=0.01)
LMR = st.sidebar.slider("The ratio of lymphocytes to mononuclear macrophages", 0.00, 10.00, value=1.98, step=0.01)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'No':0,'Yes':1,'<400':0,'≥400':1}
Solitary =map[Solitary]
Vascular_invasion = map[Vascular_invasion]
AFP_400 = map[AFP_400]

# 数据读取，特征标注
hp_train = pd.read_csv('data_m_2.csv')
hp_train['Extrahepatic_metastasis'] = hp_train['Extrahepatic_metastasis'].apply(lambda x : +1 if x==1 else 0)
features =["Solitary","Vascular_invasion","AFP_400","Age","PreAlb",'PreGGT',"PreALP",'PreFib','N','Plt','NLR','PLR','LMR']
target = 'Extrahepatic_metastasis'
random_state_new = 50
oversample = SMOTE()
# oversample = ADASYN()
X_data, y_data = oversample.fit_resample(hp_train[features],hp_train[target])
X_ros = np.array(X_data)
y_ros = np.array(y_data)
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=0.01,
                    power_t=0.5,
                    max_iter=200,
                    shuffle=True, random_state=random_state_new)
# XGB = XGBClassifier(n_estimators=360, max_depth=2, learning_rate=0.1,random_state = 0)
mlp.fit(X_ros, y_ros)
sp = 0.5
#figure
is_t = (mlp.predict_proba(np.array([[Solitary,Vascular_invasion,AFP_400,Age,PreAlb,PreGGT,PreALP,PreFib,N,Plt,NLR,PLR,LMR]]))[0][1])> sp
prob = (mlp.predict_proba(np.array([[Solitary,Vascular_invasion,AFP_400,Age,PreAlb,PreGGT,PreALP,PreFib,N,Plt,NLR,PLR,LMR]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk Metastasis'
else:
    result = 'Low Risk Metastasis'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Metastasis':
        st.balloons()
    st.markdown('## Probability of High risk Metastasis group:  '+str(prob)+'%')



