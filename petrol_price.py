
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score

# st.title("India Petrol Price Prediction")

# df = pd.read_csv("petrol_price_prediction.csv")

# st.subheader("Dataset Preview")
# st.dataframe(df)

# df['Date'] = pd.to_datetime(df['Date'])
# df['Year'] = df['Date'].dt.year
# df['Month'] = df['Date'].dt.month
# df = df.drop(['Date'], axis=1)

# X = df.drop("Petrol_Price", axis=1)
# y = df["Petrol_Price"]

# X_train, X_test, y_train, y_test = train_test_split(
# X, y, test_size=0.2, random_state=42)

# model = RandomForestRegressor()

# model.fit(X_train, y_train)

# pred = model.predict(X_test)

# accuracy = r2_score(y_test, pred)

# st.subheader("Model Accuracy")
# st.write(accuracy)

# fig, ax = plt.subplots()

# ax.scatter(y_test, pred)

# ax.set_xlabel("Actual Price")
# ax.set_ylabel("Predicted Price")

# st.pyplot(fig)

# st.subheader("Predict Petrol Price")

# crude = st.number_input("Crude Oil Price")
# dollar = st.number_input("Dollar Rate")
# demand = st.number_input("Demand Index")
# year = st.number_input("Year")
# month = st.number_input("Month")

# if st.button("Predict Price"):

#     future = [[crude, dollar, demand, year, month]]

#     prediction = model.predict(future)

#     st.success(f"Predicted Petrol Price: {prediction[0]:.2f}")




import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -------- Background Style --------
page_bg = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://images.unsplash.com/photo-1556740738-b6a63e27c4df");
background-size: cover;
background-position: center;
background-attachment: fixed;
}

h1{
color:#FFD700;
text-align:center;
}

.stButton>button{
background-color:#ff4b4b;
color:white;
border-radius:10px;
height:3em;
width:200px;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# -------- Title --------
st.title("⛽ India Petrol Price Prediction Dashboard")

# -------- Load Dataset --------
df = pd.read_csv("petrol_price_prediction.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df)

# -------- Data Processing --------
df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

df = df.drop(['Date'], axis=1)

X = df.drop("Petrol_Price", axis=1)
y = df["Petrol_Price"]

# -------- Train Test Split --------
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

# -------- ML Model --------
model = RandomForestRegressor()
model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = r2_score(y_test, pred)

st.subheader("📈 Model Accuracy")
st.success(f"Accuracy (R² Score): {accuracy:.2f}")

# -------- Scatter Plot --------
fig, ax = plt.subplots()

ax.scatter(y_test, pred, color="blue")

ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")

ax.set_title("Actual vs Predicted Petrol Price")

st.pyplot(fig)

# -------- Trend Graph --------
st.subheader("📉 Petrol Price Trend")

trend = df.groupby("Year")["Petrol_Price"].mean()

fig2, ax2 = plt.subplots()

ax2.plot(trend.index, trend.values, marker='o')

ax2.set_xlabel("Year")
ax2.set_ylabel("Average Petrol Price")

st.pyplot(fig2)

# -------- Prediction Section --------
st.subheader("🔮 Predict Petrol Price")

city = st.selectbox("Select City", [
"Lucknow","Kanpur","Varanasi","Prayagraj","Agra"
])

crude = st.number_input("Crude Oil Price")
dollar = st.number_input("Dollar Rate")
demand = st.number_input("Demand Index")
year = st.number_input("Year",2024,2035)
month = st.number_input("Month",1,12)

if st.button("Predict Petrol Price"):

    future = [[crude, dollar, demand, year, month]]

    prediction = model.predict(future)

    st.success(f"⛽ Predicted Petrol Price in {city}: ₹{prediction[0]:.2f}")






# -------- Petrol Price Map --------
st.subheader("🗺️ Uttar Pradesh Petrol Price Map")

map_data = pd.DataFrame({
    "city": ["Lucknow","Kanpur","Varanasi","Prayagraj","Agra"],
    "lat": [26.8467,26.4499,25.3176,25.4358,27.1767],
    "lon": [80.9462,80.3319,82.9739,81.8463,78.0081],
    "price": [96,95,97,96.5,95.5]
})

st.map(map_data, latitude="lat", longitude="lon")