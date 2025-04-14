import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib.pyplot as plt

# Extended training data with 15 examples per author
texts = [
    # Author A – Tech Expert
    "Artificial intelligence is reshaping the future of work.",
    "Data science unlocks new business insights every day.",
    "Machine learning enables systems to learn from experience.",
    "AI will transform industries globally in the coming decade.",
    "Neural networks are modeled after the human brain.",
    "Big data analytics is transforming modern marketing.",
    "Deep learning models are often used in image recognition.",
    "AI ethics is a crucial topic in current tech development.",
    "Natural language processing helps machines understand text.",
    "Predictive algorithms can optimize supply chains.",
    "The evolution of robotics is driven by advances in AI.",
    "Tech companies are investing heavily in AI research.",
    "AI-driven automation is changing the labor market.",
    "Voice assistants rely on natural language understanding.",
    "Computer vision enables machines to interpret the world.",

    # Author B – Legal Professional
    "Legal technology is essential for modern law firms.",
    "Court systems are slowly adapting to digital transformation.",
    "Regulatory compliance is a growing area of legal practice.",
    "Contracts must comply with national and EU legislation.",
    "Forensic linguistics can assist in authorship attribution.",
    "New privacy laws require stricter data governance.",
    "Due process must be upheld in all legal proceedings.",
    "The judiciary must remain independent and impartial.",
    "Legal scholars are debating the use of AI in courts.",
    "Digital evidence must be authenticated to be admissible.",
    "Administrative law governs public sector decision-making.",
    "There is a fine balance between security and liberty.",
    "The constitution protects fundamental rights and freedoms.",
    "International law affects national sovereignty in complex ways.",
    "The rule of law underpins a functioning democracy.",

    # Author C – Conversational Style
    "Hey, just wanted to check if you're free this weekend?",
    "Can't believe how good that new show is – totally obsessed!",
    "I'm running late but grabbing coffee on the way!",
    "Honestly, I'm just so tired lately. Life's been a lot.",
    "LOL that was wild, we have to do that again sometime.",
    "Do you wanna hang out later or maybe grab dinner?",
    "Omg I forgot to reply – my brain is a mess right now.",
    "Guess what happened today?! You're not gonna believe it.",
    "I'm lowkey freaking out but trying to act normal.",
    "That was such a vibe, I miss those chill days.",
    "Just checking in – how are you holding up?",
    "Ugh, Monday again. I need coffee and silence.",
    "Wanna binge something dumb and forget the world?",
    "Lmk if you wanna vent or just talk – I'm here.",
    "This week has been chaos. I’m running on vibes only."
]
labels = ["Author A"] * 15 + ["Author B"] * 15 + ["Author C"] * 15

# Train the model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
clf = MultinomialNB()
clf.fit(X, labels)

# Streamlit UI
st.title("Linguistic Fingerprint Identifier")

st.markdown("""
**Author Profiles:**
- 👤 **Author A – Tech Expert**: Focused on AI, data science, and futuristic technology. Language is factual, visionary, and technical.
- 👩‍⚖️ **Author B – Legal Professional**: Writes about law, compliance, and justice systems. Formal tone with legal vocabulary.
- 🧍‍♀️ **Author C – Conversational**: Informal, personal style like chats, texts, and social media posts.
""")

st.write("Paste a text and see which author's style it resembles most.")

input_text = st.text_area("Your text to analyze:", height=200)

if st.button("Analyze") and input_text.strip():
    X_new = vectorizer.transform([input_text])
    proba = clf.predict_proba(X_new)[0]
    pred_author = clf.classes_[np.argmax(proba)]
    confidence = np.max(proba) * 100

    st.success(f"Predicted Author: {pred_author} ({confidence:.2f}% match)")

    # Show probability bar chart
    st.write("### Match Probability")
    fig, ax = plt.subplots()
    ax.bar(clf.classes_, proba * 100)
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)
