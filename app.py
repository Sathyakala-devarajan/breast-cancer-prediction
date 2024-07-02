import streamlit as st
import pickle
import pandas as pd

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# print(model.predict([[17.99,	10.38,	122.8,	1001,	0.1184,	0.2776,	0.3001,	
#                     0.1471,	0.2419,	0.07871,	1.095,	0.9053,	8.589,	153.4,
#                     0.006399,	0.04904,	0.05373,	0.01587,	0.03003,	
#                     0.006193,	25.38,	17.33,	184.6,	2019,	0.1622,	0.6656,	
#                     0.7119,	0.2654,	0.4601,	0.1189
# ]]))

# create streamlit application
st.title('Breat cancer prediction')
mr = st.number_input('mean radius') 
mt = st.number_input('mean texture')
mp = st.number_input('mean perimeter')
ma = st.number_input('mean area')
ms = st.number_input('mean smoothness')
mcom = st.number_input('mean compactness')
mc = st.number_input('mean concavity')
mcp = st.number_input('mean concave points')
msy = st.number_input('mean symmetry')
mfd = st.number_input('mean fractal dimension')
re = st.number_input('radius error')
te = st.number_input('texture error')
pe = st.number_input('perimeter error')
ae = st.number_input('area error')
se = st.number_input('smoothness error')
come = st.number_input('compactness error')
ce = st.number_input('concavity error')	
cpe = st.number_input('concave points error')
sye = st.number_input('symmetry error')
fde = st.number_input('fractal dimension error')
wr = st.number_input('worst radius')
wt = st.number_input('worst texture')
wp = st.number_input('worst perimeter')
wa = st.number_input('worst area')
ws = st.number_input('worst smoothness')
wcom = st.number_input('worst compactness')
wc = st.number_input('worst concavity')
wcp = st.number_input('worst concave points')
wsy = st.number_input('worst symmetry')
wfd = st.number_input('worst fractal dimension')

button = st.button('Predict')

if button:
    result = model.predict([[mr, mt, mp, ma, ms, mcom, mc, mcp, msy, mfd,
                             re, te, pe, ae, se, come, ce, cpe, sye, fde,
                             wr, wt, wp, wa, ws, wcom, wc, wcp, wsy, wfd
                             ]])
    st.write(f"The predicted result is {result[0]}")
