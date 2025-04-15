import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib.pyplot as plt
import PyPDF2
import re

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
**About the Application – Linguistic Fingerprint Identifier**  
This tool enables the analysis of written language to identify stylistic patterns that may be characteristic of a particular type of author. Using natural language processing and machine learning, the system estimates the likelihood that a given text corresponds to one of the predefined author profiles.  

The application may be of interest in legal, forensic, or regulatory contexts where authorship attribution, stylistic analysis, or text comparison is of relevance. Please note that this is a demonstrational prototype; analytical results should be interpreted with caution and not be considered conclusive evidence.  

**Available Author Profiles:**  
- **Author A:** Technical expert in AI and data science  
- **Author B:** Legal professional with formal writing style  
- **Author C:** Informal, conversational style typical of everyday communication
""")

conversation_text = st.text_area("Paste a conversation between two people (use 'Person 1:' and 'Person 2:' to mark their lines):", height=300)

if st.button("Analyze Conversation") and conversation_text.strip():
    # Separate text per person
    person1_lines = re.findall(r"Person 1:(.*)", conversation_text)
    person2_lines = re.findall(r"Person 2:(.*)", conversation_text)

    text1 = " ".join(person1_lines).strip()
    text2 = " ".join(person2_lines).strip()

    if text1:
        X1 = vectorizer.transform([text1])
        proba1 = clf.predict_proba(X1)[0]
        pred1 = clf.classes_[np.argmax(proba1)]
        conf1 = np.max(proba1) * 100
        st.markdown(f"**Person 1 → {pred1} ({conf1:.2f}% match)**")

        fig1, ax1 = plt.subplots()
        ax1.bar(clf.classes_, proba1 * 100)
        ax1.set_ylabel("Probability (%)")
        ax1.set_title("Person 1")
        ax1.set_ylim(0, 100)
        st.pyplot(fig1)

    if text2:
        X2 = vectorizer.transform([text2])
        proba2 = clf.predict_proba(X2)[0]
        pred2 = clf.classes_[np.argmax(proba2)]
        conf2 = np.max(proba2) * 100
        st.markdown(f"**Person 2 → {pred2} ({conf2:.2f}% match)**")

        fig2, ax2 = plt.subplots()
        ax2.bar(clf.classes_, proba2 * 100)
        ax2.set_ylabel("Probability (%)")
        ax2.set_title("Person 2")
        ax2.set_ylim(0, 100)
        st.pyplot(fig2)

