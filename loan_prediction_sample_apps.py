import streamlit as st
import pickle
import pandas as pd

df=pd.read_csv('loan_approval_dataset.csv')
df.columns=df.columns.str.strip()

def load_model():
    with open('loan_prediction.pkl','rb') as f:
        return pickle.load(f)


st.header("Loan Prediction Apps")

with st.sidebar:
    st.image('parami.jpg')
    st.text("Student Name: John")
    st.text("Student ID: PU....")
    st.text("Class of 2027")


col1,col2=st.columns(2)
with col1:

    var1=st.number_input("Enter no of dependents",
                         max_value=int(df['no_of_dependents'].max()),
                         min_value=int(df['no_of_dependents'].min()),
                         value=4)
    var2=st.selectbox("Enter your education",df['education'].unique())
    var3=st.selectbox("Enter your employment status",df['self_employed'].unique())
    var4=st.slider("Enter your annual income",
                   max_value=df['income_annum'].max(),
                   min_value=df['income_annum'].min(),step=100)
    var5=st.slider("Enter your loan term",
                   max_value=df['loan_term'].max(),
                   min_value=df['loan_term'].min(),step=2)

with col2:
    var6=st.slider("Enter your cibil_score ",
                   max_value=df['cibil_score'].max(),
                   min_value=df['cibil_score'].min(),step=20) 
    var7=st.slider("Enter your residential_assets_value ",
                   max_value=df['residential_assets_value'].max(),
                   min_value=df['residential_assets_value'].min(),step=1000)
    var8=st.slider("Enter your commercial_assets_value ",
                   max_value=df['commercial_assets_value'].max(),
                   min_value=df['commercial_assets_value'].min(),step=1000)
    var9=st.slider("Enter your luxury_assets_value ",
                   max_value=df['luxury_assets_value'].max(),
                   min_value=df['luxury_assets_value'].min(),step=2000)
    var10=st.slider("Enter your bank_asset_value ",
                   max_value=df['bank_asset_value'].max(),
                   min_value=df['bank_asset_value'].min(),step=500)

predict_button=st.button("Predict Loan Amount")


if predict_button:
    input_data=pd.DataFrame(
            [
                {
                    "no_of_dependents":var1,
                    "education":var2,
                    "self_employed":var3,
                    "income_annum":var4,
                    "loan_term":var5,
                    "cibil_score":var6,
                    "residential_assets_value":var7,
                    "commercial_assets_value":var8,
                    "luxury_assets_value":var9,
                    "bank_asset_value":var10
                }
            ]
        )

    model=load_model()
    result=model.predict(input_data)[0]
    st.write(f"Loan Amount: {result}")


