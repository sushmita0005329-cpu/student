import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Page title
st.title("📚 Library Frequent User Prediction")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("library_data.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Show group statistics
st.subheader("Average Values by Frequent User")
st.write(df.groupby("FrequentUser").mean())

# Features and target
X = df[["StudentAge", "BooksIssued", "LateReturns", "MembershipYears"]]
y = df["FrequentUser"]

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

st.subheader("Model Accuracy")
st.success(f"Accuracy: {accuracy * 100:.2f}%")

# ---- User Input Section ----
st.subheader("🔍 Predict Frequent User")

age = st.number_input("Student Age", min_value=5, max_value=100, value=10)
books = st.number_input("Books Issued", min_value=0, max_value=50, value=1)
late = st.number_input("Late Returns", min_value=0, max_value=20, value=3)
years = st.number_input("Membership Years", min_value=0, max_value=20, value=4)

if st.button("Predict"):
    new_data = [[age, books, late, years]]
    prediction = model.predict(new_data)

    if prediction[0] == 1:
        st.error("❌ Frequent User : NO")
    else:
        st.success("✅ Frequent User : YES")



# import streamlit as st
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# # ---- Page Config ----
# st.set_page_config(page_title="Library Frequent User Prediction", layout="centered")

# # ---- Background Image ----
# page_bg_img = """
# <style>
# [data-testid="stAppViewContainer"] {
#     background-image: url("https://images.unsplash.com/photo-1512820790803-83ca734da794");
#     background-size: cover;
#     background-position: center;
#     background-repeat: no-repeat;
# }
# [data-testid="stHeader"] {
#     background: rgba(0,0,0,0);
# }
# [data-testid="stToolbar"] {
#     right: 2rem;
# }
# </style>
# """
# st.markdown(page_bg_img, unsafe_allow_html=True)

# # ---- Page Title ----
# st.title("📚 Library Frequent User Prediction")

# # ---- Load Dataset ----
# @st.cache_data
# def load_data():
#     return pd.read_csv("library_data.csv")

# df = load_data()

# st.subheader("Dataset Preview")
# st.dataframe(df.head())

# # ---- Group Statistics ----
# st.subheader("Average Values by Frequent User")
# st.write(df.groupby("FrequentUser").mean())

# # ---- Features and Target ----
# X = df[["StudentAge", "BooksIssued", "LateReturns", "MembershipYears"]]
# y = df["FrequentUser"]

# # Encode target
# le = LabelEncoder()
# y_enc = le.fit_transform(y)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_enc, test_size=0.2, random_state=42
# )

# # Train model
# model = DecisionTreeClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Evaluate model
# pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, pred)

# st.subheader("Model Accuracy")
# st.success(f"Accuracy: {accuracy * 100:.2f}%")

# # ---- User Input Section ----
# st.subheader("🔍 Predict Frequent User")

# age = st.number_input("Student Age", min_value=5, max_value=100, value=10)
# books = st.number_input("Books Issued", min_value=0, max_value=50, value=1)
# late = st.number_input("Late Returns", min_value=0, max_value=20, value=3)
# years = st.number_input("Membership Years", min_value=0, max_value=20, value=4)

# if st.button("Predict"):
#     new_data = [[age, books, late, years]]
#     prediction = model.predict(new_data)

#     if prediction[0] == 1:
#         st.error("❌ Frequent User : NO")
#     else:
#         st.success("✅ Frequent User : YES")

