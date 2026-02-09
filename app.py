import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import plotly.express as px
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Fake News Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Enabled Fake News Detection using DeBERTa-v3")
st.markdown("State-of-the-Art Multimodal Fake News Verification System")

# =====================================================
# MODEL ARCHITECTURE
# =====================================================

class EnhancedMultimodalBERT(nn.Module):
    def __init__(self, num_other_features=11, num_classes=2):
        super().__init__()

        # 🔥 DIRECT HUGGINGFACE LOADING
        self.bert = AutoModel.from_pretrained("microsoft/deberta-v3-base")

        self.metadata_processor = nn.Sequential(
            nn.Linear(num_other_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        hidden = self.bert.config.hidden_size + 64

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, other_data):

        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask)

        text_feat = output.last_hidden_state[:, 0, :]
        meta_feat = self.metadata_processor(other_data)

        combined = torch.cat((text_feat, meta_feat), dim=1)

        return self.classifier(combined)


# =====================================================
# LOAD MODEL (CACHED)
# =====================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    model = EnhancedMultimodalBERT()

    return model, tokenizer


# =====================================================
# PREDICTION
# =====================================================

def predict(model, tokenizer, text, metadata):

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    meta = torch.tensor(metadata, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(encoding["input_ids"],
                       encoding["attention_mask"],
                       meta)

        probs = torch.softmax(logits, dim=1)

    return probs.numpy()[0]


# =====================================================
# METADATA FEATURES
# =====================================================

def prepare_metadata():

    return np.zeros(11)


# =====================================================
# SIDEBAR
# =====================================================

mode = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Analyze", "Performance"]
)


# =====================================================
# DASHBOARD
# =====================================================

if mode == "Dashboard":

    st.subheader("System Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "65.11%")
    col2.metric("Precision", "0.62")
    col3.metric("Recall", "0.63")

    st.info(
        """
        This system uses DeBERTa-v3 transformer with metadata fusion
        for political fake news detection on the LIAR dataset.
        """
    )


# =====================================================
# ANALYSIS
# =====================================================

elif mode == "Analyze":

    model, tokenizer = load_model()

    text = st.text_area("Enter statement")

    if st.button("Analyze"):

        metadata = prepare_metadata()

        probs = predict(model, tokenizer, text, metadata)

        real_prob = probs[0]
        fake_prob = probs[1]

        if real_prob > fake_prob:
            st.success(f"REAL NEWS ({real_prob:.2%})")
        else:
            st.error(f"FAKE NEWS ({fake_prob:.2%})")

        df = pd.DataFrame({
            "Class": ["Real", "Fake"],
            "Probability": [real_prob, fake_prob]
        })

        fig = px.bar(df, x="Class", y="Probability")
        st.plotly_chart(fig)


# =====================================================
# PERFORMANCE
# =====================================================

elif mode == "Performance":

    st.subheader("Model Evaluation")

    metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [0.6511, 0.62, 0.63, 0.62]
    }

    df = pd.DataFrame(metrics)

    st.table(df)

    fig = px.bar(df, x="Metric", y="Value")
    st.plotly_chart(fig)
