import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------- LOAD DATA --------------------
df = pd.read_csv("spam.csv", encoding='latin-1')

# Select only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# -------------------- CLEAN DATA --------------------
df['message'] = df['message'].astype(str)
df = df[df['message'].str.strip() != ""]

df['label'] = df['label'].astype(str).str.strip().str.lower()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.dropna()

# -------------------- UI TITLE --------------------
st.title("📩 Spam Message Detector")

# -------------------- GRAPH --------------------
st.subheader("📊 Dataset Overview")

counts = df['label'].value_counts()
spam_count = counts.get(1, 0)
ham_count = counts.get(0, 0)

fig, ax = plt.subplots()
ax.bar(['Ham (Not Spam)', 'Spam'], [ham_count, spam_count])
ax.set_title("Spam vs Not Spam Messages")
ax.set_xlabel("Message Type")
ax.set_ylabel("Count")

st.pyplot(fig)

# -------------------- MODEL --------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# -------------------- USER INPUT --------------------
st.subheader("📩 Check Your Message")

user_input = st.text_area("Enter your message:")

if st.button("Check"):
    if user_input.strip() != "":
        msg_vec = vectorizer.transform([user_input])
        prediction = model.predict(msg_vec)

        if prediction[0] == 1:
            st.error("🚨 This is a SPAM message!")
        else:
            st.success("✅ This is NOT a Spam message")
    else:
        st.warning("⚠️ Please enter a message")