import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
from LVModel import LVModel
from streamlit_extras.card import card
from streamlit_extras.metric_cards import style_metric_cards

# Set environment variables for debugging and parallelism
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the model and tokenizer
@st.cache_resource
def load_model():
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model = LVModel(base_model=base_model)

    # Load pretrained weights (adjust path to your saved model)
    state_dict = torch.load("./t5_lv_model/FlanT5.bin", map_location=device)
    encoder_embed_weight = state_dict["base_model.encoder.embed_tokens.weight"]
    state_dict["base_model.shared.weight"] = encoder_embed_weight
    state_dict["base_model.decoder.embed_tokens.weight"] = encoder_embed_weight
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    return model, tokenizer


# Prepare data for training
def prepare_data_for_training(data, tokenizer, max_length=128):
    inputs = []
    labels = []
    label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}

    for example in data:
        premise = example["input"]["premise"]
        hypothesis = example["input"]["hypothesis"]
        label = example["output"]

        input_text = f"{premise} <sep> {hypothesis}"
        encoding = tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        if label in label_map:
            inputs.append(encoding["input_ids"])
            labels.append(label_map[label])
        else:
            print(f"Invalid label: {label}. Skipping example.")

    inputs_tensor = torch.cat(inputs, dim=0) if inputs else torch.empty(0)
    labels_tensor = torch.tensor(labels) if labels else torch.empty(0, dtype=torch.long)

    return inputs_tensor, labels_tensor


# Streamlit UI
st.set_page_config(
    page_title="AI Text Analysis",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS with pixel art theme
st.markdown(
    """
    <style>
    /* Dark pixel theme */
    .stApp {
        background-color: #2B2D42;  /* Deep navy blue */
        color: #EDF2F4;  /* Soft white */
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1A1B2E;  /* Darker navy */
        border-right: 2px solid #8D99AE;  /* Pixel border */
    }
    
    /* Input containers */
    .input-container {
        background-color: #1A1B2E;
        padding: 2rem;
        border-radius: 8px;  /* Less rounded for pixel look */
        border: 2px solid #8D99AE;
        box-shadow: 4px 4px 0px #EF233C;  /* Pixel shadow */
        margin-bottom: 1.5rem;
    }
    
    /* Results container */
    .results-container {
        background-color: #1A1B2E;
        padding: 2rem;
        border-radius: 8px;
        border: 2px solid #8D99AE;
        box-shadow: 4px 4px 0px #EF233C;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #8D99AE;
    }
    .stProgress .st-bo {
        background-color: #D90429;  /* Bright red */
        height: 24px;
        border-radius: 4px;
    }
    
    /* Text areas */
    .stTextArea textarea {
        background-color: #2B2D42 !important;
        color: #EDF2F4 !important;
        border: 2px solid #8D99AE !important;
        font-size: 1.1rem !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #D90429 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #EDF2F4 !important;
    }
    
    /* Cards */
    .card {
        background-color: #1A1B2E !important;
        border: 2px solid #8D99AE !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #D90429 !important;
        color: #EDF2F4 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 4px !important;
        border: 2px solid #8D99AE !important;
        box-shadow: 3px 3px 0px #8D99AE !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1A1B2E;
        border-radius: 4px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #2B2D42;
        border-radius: 4px;
        color: #EDF2F4;
        font-size: 1rem;
    }
    
    /* Prediction labels */
    .prediction-high {
        color: #4DAA57;  /* Pixel green */
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction-medium {
        color: #FFD93D;  /* Pixel yellow */
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction-low {
        color: #D90429;  /* Pixel red */
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png")
    st.title("Text Analysis")
    st.markdown(
        """
        This tool analyzes the logical relationship between two pieces of text 
        using advanced AI technology.
    """
    )

    with st.expander("ℹ️ How to use"):
        st.markdown(
            """
            1. Enter a premise (base statement)
            2. Enter a hypothesis (statement to verify)
            3. Click 'Analyze' to see the relationship
            
            The AI will classify the relationship as:
            - **Entailment**: Hypothesis follows from premise
            - **Contradiction**: Hypothesis conflicts with premise
            - **Neutral**: No logical connection
        """
        )

    st.markdown("---")
    st.markdown("Made with ❤️ using Flan-T5 by Ly Ngoc Vu")

# Main content
st.title("Natural Language Inference")

# Input section
st.markdown("### Input Statements")
with st.container():
    tab1, tab2 = st.tabs(["📝 Text Input", "ℹ️ Examples"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            with st.container():
                st.markdown(
                    """
                    <div class='input-container'>
                    <h4>Premise</h4>
                """,
                    unsafe_allow_html=True,
                )
                premise = st.text_area(
                    "",
                    placeholder="Enter the base statement...",
                    height=100,
                    key="premise_input",
                    help="This is the foundation statement that will be analyzed.",
                )
                st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            with st.container():
                st.markdown(
                    """
                    <div class='input-container'>
                    <h4>Hypothesis</h4>
                """,
                    unsafe_allow_html=True,
                )
                hypothesis = st.text_area(
                    "",
                    placeholder="Enter the statement to verify...",
                    height=100,
                    key="hypothesis_input",
                    help="This statement will be compared against the premise.",
                )
                st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown(
            """
            **Example 1:**
            - Premise: The cat is sleeping on the windowsill.
            - Hypothesis: There is a cat.
            
            **Example 2:**
            - Premise: The restaurant is busy tonight.
            - Hypothesis: The restaurant is closed.
        """
        )

# Analysis button
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

# Results section
if analyze:
    if not premise or not hypothesis:
        st.error("Please provide both premise and hypothesis.")
    else:
        model, tokenizer = load_model()

        with st.spinner("🤔 Analyzing relationship..."):
            inputs = tokenizer(
                f"{premise} <sep> {hypothesis}",
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            decoder_input_ids = torch.full(
                (1, 1),
                model.base_model.config.decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                )
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1).item()
                confidence = probs[0][prediction].item()

                label_map = {0: "Entailment", 1: "Contradiction", 2: "Neutral"}
                st.markdown("### Analysis Results")
                with st.container():
                    st.markdown(
                        '<div class="results-container">', unsafe_allow_html=True
                    )

                    # Metrics row
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Prediction",
                            label_map[prediction],
                            delta=(
                                "High Confidence"
                                if confidence > 0.8
                                else "Medium Confidence"
                            ),
                        )
                    with col2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    with col3:
                        st.metric("Processing Time", "< 1 sec")
                    style_metric_cards()

                    # Probability distribution
                    st.markdown("#### Probability Distribution")
                    for idx, label in label_map.items():
                        prob = probs[0][idx].item()
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.progress(prob)
                        with col2:
                            confidence_class = (
                                "prediction-high"
                                if prob > 0.8
                                else (
                                    "prediction-medium"
                                    if prob > 0.5
                                    else "prediction-low"
                                )
                            )
                            st.markdown(
                                f"""
                                <span class="{confidence_class}">
                                    {label}: {prob:.1%}
                                </span>
                            """,
                                unsafe_allow_html=True,
                            )

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Explanation
                    if prediction == 0:
                        st.success(
                            "✅ The hypothesis logically follows from the premise."
                        )
                    elif prediction == 1:
                        st.error("❌ The hypothesis contradicts the premise.")
                    else:
                        st.info(
                            "ℹ️ The hypothesis neither follows from nor contradicts the premise."
                        )

# Footer
# Footer
st.markdown("---")
st.markdown(
    """
    <div style='background-color: #1e293b; padding: 2rem; border-radius: 12px; border: 1px solid #334155;'>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    card(
        title="📊 Fast Analysis",
        text="Advanced optimization for real-time results",
        image="https://img.icons8.com/fluency/48/000000/fast-forward.png",
        styles={
            "card": {
                "background-color": "#334155",
                "border": "none",
                "padding": "1.5rem",
            }
        },
    )
with col2:
    card(
        title="🤖 AI Powered",
        text="Leveraging Flan-T5 for precise inference",
        image="https://img.icons8.com/fluency/48/000000/artificial-intelligence.png",
        styles={
            "card": {
                "background-color": "#334155",
                "border": "none",
                "padding": "1.5rem",
            }
        },
    )
with col3:
    card(
        title="🎯 High Accuracy",
        text="Fine-tuned on NLI datasets for optimal performance",
        image="https://img.icons8.com/fluency/48/000000/accuracy.png",
        styles={
            "card": {
                "background-color": "#334155",
                "border": "none",
                "padding": "1.5rem",
            }
        },
    )

st.markdown("</div>", unsafe_allow_html=True)

# Version and contact info
st.markdown(
    """
    <div style='text-align: center; margin-top: 2rem; color: #94a3b8; font-size: 0.875rem;'>
        v1.0.0 | <a href="https://github.com/zhugez" style="color: #60a5fa; text-decoration: none;">GitHub</a> | 
        <a href="mailto:dezzhuge@gmail.com" style="color: #60a5fa; text-decoration: none;">Contact</a>
    </div>
    """,
    unsafe_allow_html=True,
)
