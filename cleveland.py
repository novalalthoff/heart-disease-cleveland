import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import time

# Column names
columns = [
  'age', 'sex', 'cp', 'trestbps', 'chol',
  'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
  'slope', 'ca', 'thal', 'num'
]

# Read dataset
df = pd.read_csv("processed.cleveland.data", names=columns, header=None)

# Replacing '?' with 'NaN'
df.replace('?', np.nan, inplace=True)

updated_df = df.dropna()

# Change the data type for each column
for column in updated_df.columns:
  if updated_df[column].dtypes == 'O':
    updated_df[column] = updated_df[column].astype("float64")

features = updated_df.drop('num', axis=1)
target = updated_df['num']

# Oversampling target 0, 2, 3, 4 based on target 1
oversample = SMOTE(k_neighbors=4)
X, y = oversample.fit_resample(features, target)

filled_df = X
filled_df['num'] = y

features = filled_df.drop('num', axis=1)
target = filled_df['num']

# Train - Test split
X_train ,X_test, y_train ,y_test = train_test_split(features, target, test_size = 0.2)

model = DecisionTreeClassifier()

accuracy_list = np.array([])

for i in range(0, 10):
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  accuracy = round((accuracy * 100), 2)

  accuracy_list = np.append(accuracy_list, accuracy)

min_accuracy = np.min(accuracy_list)
max_accuracy = np.max(accuracy_list)

# ========================================================================================================================================================================================

# STREAMLIT
st.set_page_config(
  page_title = "Cleveland Heart Disease",
  page_icon = ":heart:"
)

st.title("Cleveland Heart Disease")
st.write("_Using Decision Tree Classifier_")
st.write(f"**Model's Accuracy**: :red[**{min_accuracy}%**] - :orange[**{max_accuracy}%**] (_Needs improvement! :red[Do not copy outright]_)")
st.write("")

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

with tab1:
  st.sidebar.header("**User Input** Sidebar")

  age = st.sidebar.number_input(label=":violet[**Age**]", min_value=filled_df['age'].min(), max_value=filled_df['age'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{filled_df['age'].min()}**], :red[Max] value: :red[**{filled_df['age'].max()}**]")
  st.sidebar.write("")

  sex_sb = st.sidebar.selectbox(label=":violet[**Sex**]", options=["Male", "Female"])
  st.sidebar.write("")
  st.sidebar.write("")
  if sex_sb == "Male":
    sex = 1
  elif sex_sb == "Female":
    sex = 0
  # -- Value 0: Female
  # -- Value 1: Male

  cp_sb = st.sidebar.selectbox(label=":violet[**Chest pain type**]", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
  st.sidebar.write("")
  st.sidebar.write("")
  if cp_sb == "Typical angina":
    cp = 1
  elif cp_sb == "Atypical angina":
    cp = 2
  elif cp_sb == "Non-anginal pain":
    cp = 3
  elif cp_sb == "Asymptomatic":
    cp = 4
  # -- Value 1: typical angina
  # -- Value 2: atypical angina
  # -- Value 3: non-anginal pain
  # -- Value 4: asymptomatic

  trestbps = st.sidebar.number_input(label=":violet[**Resting blood pressure** (in mm Hg on admission to the hospital)]", min_value=filled_df['trestbps'].min(), max_value=filled_df['trestbps'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{filled_df['trestbps'].min()}**], :red[Max] value: :red[**{filled_df['trestbps'].max()}**]")
  st.sidebar.write("")

  chol = st.sidebar.number_input(label=":violet[**Serum cholestoral** (in mg/dl)]", min_value=filled_df['chol'].min(), max_value=filled_df['chol'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{filled_df['chol'].min()}**], :red[Max] value: :red[**{filled_df['chol'].max()}**]")
  st.sidebar.write("")

  fbs_sb = st.sidebar.selectbox(label=":violet[**Fasting blood sugar > 120 mg/dl?**]", options=["False", "True"])
  st.sidebar.write("")
  st.sidebar.write("")
  if fbs_sb == "False":
    fbs = 0
  elif fbs_sb == "True":
    fbs = 1
  # -- Value 0: false
  # -- Value 1: true

  restecg_sb = st.sidebar.selectbox(label=":violet[**Resting electrocardiographic results**]", options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
  st.sidebar.write("")
  st.sidebar.write("")
  if restecg_sb == "Normal":
    restecg = 0
  elif restecg_sb == "Having ST-T wave abnormality":
    restecg = 1
  elif restecg_sb == "Showing left ventricular hypertrophy":
    restecg = 2
  # -- Value 0: normal
  # -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST  elevation or depression of > 0.05 mV)
  # -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

  thalach = st.sidebar.number_input(label=":violet[**Maximum heart rate achieved**]", min_value=filled_df['thalach'].min(), max_value=filled_df['thalach'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{filled_df['thalach'].min()}**], :red[Max] value: :red[**{filled_df['thalach'].max()}**]")
  st.sidebar.write("")

  exang_sb = st.sidebar.selectbox(label=":violet[**Exercise induced angina?**]", options=["No", "Yes"])
  st.sidebar.write("")
  st.sidebar.write("")
  if exang_sb == "No":
    exang = 0
  elif exang_sb == "Yes":
    exang = 1
  # -- Value 0: No
  # -- Value 1: Yes

  oldpeak = st.sidebar.number_input(label=":violet[**ST depression induced by exercise relative to rest**]", min_value=filled_df['oldpeak'].min(), max_value=filled_df['oldpeak'].max())
  st.sidebar.write(f":orange[Min] value: :orange[**{filled_df['oldpeak'].min()}**], :red[Max] value: :red[**{filled_df['oldpeak'].max()}**]")
  st.sidebar.write("")

  slope_sb = st.sidebar.selectbox(label=":violet[**The slope of the peak exercise ST segment**]", options=["Upsloping", "Flat", "Downsloping"])
  st.sidebar.write("")
  st.sidebar.write("")
  if slope_sb == "Upsloping":
    slope = 1
  elif slope_sb == "Flat":
    slope = 2
  elif slope_sb == "Downsloping":
    slope = 3
  # -- Value 1: upsloping
  # -- Value 2: flat
  # -- Value 3: downsloping

  ca = st.sidebar.select_slider(":violet[**Number of major vessels** (0-3) colored by flourosopy]", options=[0, 1, 2, 3])
  st.sidebar.write("")

  thal_sb = st.sidebar.selectbox(label=":violet[**Thal**]", options=["Normal", "Fixed defect", "Reversable defect"])
  st.sidebar.write("")
  st.sidebar.write("")
  if thal_sb == "Normal":
    thal = 3
  elif thal_sb == "Fixed defect":
    thal = 6
  elif thal_sb == "Reversable defect":
    thal = 7
  # -- Value 3: normal
  # -- Value 6: fixed defect
  # -- Value 7: reversable defect

  data = {
    'Age': age,
    'Sex': sex_sb,
    'Chest pain type': cp_sb,
    'RPB': f"{trestbps} mm Hg",
    'Serum Cholestoral': f"{chol} mg/dl",
    'FBS > 120 mg/dl?': fbs_sb,
    'Resting ECG': restecg_sb,
    'Maximum heart rate': thalach,
    'Exercise induced angina?': exang_sb,
    'ST depression': oldpeak,
    'Slope of peak exercise': slope_sb,
    'Number of major vessels': ca,
    'Thal': thal_sb,
  }

  preview_df = pd.DataFrame(data, index=['input'])

  st.header("User Input as DataFrame")
  st.write("")
  st.dataframe(preview_df.iloc[:, :5])
  st.write("")
  st.dataframe(preview_df.iloc[:, 5:9])
  st.write("")
  st.dataframe(preview_df.iloc[:, 9:])
  st.write("")

  result = ":violet[-]"

  predict_btn = st.button("**Predict**", type="primary")

  st.write("")
  if predict_btn:
    inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    prediction = model.predict(inputs)[0]

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    if prediction == 0:
      result = ":green[**Healthy**]"
    elif prediction == 1:
      result = ":orange[**Heart disease level 1**]"
    elif prediction == 2:
      result = ":orange[**Heart disease level 2**]"
    elif prediction == 3:
      result = ":red[**Heart disease level 3**]"
    elif prediction == 4:
      result = ":red[**Heart disease level 4**]"

  st.write("")
  st.write("")
  st.subheader("Prediction:")
  st.subheader(result)

with tab2:
  st.header("Predict multiple data:")

  sample_csv = filled_df.iloc[:5, :-1].to_csv(index=False).encode('utf-8')

  st.write("")
  st.download_button("Download CSV Example", data=sample_csv, file_name='sample_heart_disease_parameters.csv', mime='text/csv')

  st.write("")
  st.write("")
  file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

  if file_uploaded:
    uploaded_df = pd.read_csv(file_uploaded)
    prediction_arr = model.predict(uploaded_df)

    bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 70):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)

    result_arr = []

    for prediction in prediction_arr:
      if prediction == 0:
        result = "Healthy"
      elif prediction == 1:
        result = "Heart disease level 1"
      elif prediction == 2:
        result = "Heart disease level 2"
      elif prediction == 3:
        result = "Heart disease level 3"
      elif prediction == 4:
        result = "Heart disease level 4"
      result_arr.append(result)

    uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

    for i in range(70, 101):
      status_text.text(f"{i}% complete")
      bar.progress(i)
      time.sleep(0.01)
      if i == 100:
        time.sleep(1)
        status_text.empty()
        bar.empty()

    col1, col2 = st.columns([1, 2])

    with col1:
      st.dataframe(uploaded_result)
    with col2:
      st.dataframe(uploaded_df)
