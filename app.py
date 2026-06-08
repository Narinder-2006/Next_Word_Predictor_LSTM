import streamlit as st
import time
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
# --- PAGE CONFIG ---
st.set_page_config(
    page_title="LSTM Language Model",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>
    /* Remove default Streamlit top padding and push content flush left */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
        padding-left: 2rem !important;
        max-width: 850px !important;
        margin-left: 0px !important; /* Forces layout flush left next to sidebar */
    }

    /* Remove gap above the title injected by Streamlit's header */
    header[data-testid="stHeader"] {
        height: 0rem !important;
        visibility: hidden;
    }

    /* Tighten vertical spacing between all elements */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0 !important;
    }

    /* Style the suggestion cards */
    div[data-testid="stInfo"] {
        padding: 0.75rem 1rem !important;
        border-radius: 10px !important;
    }

    /* Remove excess spacing after <hr> markdown injections */
    .element-container:has(hr) {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }

    /* Compact sidebar layout */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
    }

    /* --- LIFT CHAT BOX INPUT UPWARD --- */
    div[data-testid="stChatInputContainer"] {
        bottom: 700px !important; /* Lifts the absolute fixed input frame higher */
    }
    
    /* --- PUSH BOTTOM MESSAGES UP (MARGIN BOTTOM) --- */
    .stChatMessageContainer {
        margin-bottom: 10rem !important; /* Keeps logs clear of the lifted input bar */
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_production_pipeline():
    
    with st.spinner("⚡ Initializing Deep Learning Core... Streaming weights into VRAM..."):
        # Load vocabulary configuration
        with open("vocab_config.json", "r") as f:
            saved_vocab = json.load(f)
        vectorizer = tf.keras.layers.TextVectorization(vocabulary=saved_vocab)
        
        # Load the model weights cleanly
        with tf.device('/GPU:0'):
                
                # 1. Directly load the entire model file (Structure + Optimizer state + Weights)
           model_v3 = load_model("models/daily_dialog_lstm_v2.keras")
           model_v2 = load_model("models/daily_dialog_lstm_v1.keras")
        print("✔ Entire .keras model loaded directly into GPU memory!")
        
    return vectorizer, model_v2,model_v3

# --- 2. EXECUTE THE PIPELINE INSTANTLY ---
# Streamlit calls this, matches the cache, and skips the spinner on future clicks!
vectorize_layer, model_v2,model_v3 = load_production_pipeline()



# --- SIDEBAR ---
st.sidebar.title("⚙️ Model Settings")
st.sidebar.markdown("Fine-tune your sequence-generation hyper-parameters below.")

temperature = st.sidebar.slider(
    "Creativity Temperature",
    min_value=0.1,
    max_value=1.5,
    value=0.5,
    step=0.1,
    help="Lower = predictable. Higher = creative/random."
)

words_to_predict = st.sidebar.number_input(
    "Words to Generate",
    min_value=1,
    max_value=100,
    value=12,
    step=1
)

model_variant = st.sidebar.radio(
    "Select Model Variant",
    options=["Model V1 (Pure Recurrent)", "Model V2 (Padded & Upgraded)"],
    index=0,
    help="Choose which trained network weights will evaluate your prompt."
)

st.sidebar.markdown("---")
st.sidebar.caption("🤖 **Architecture:** Dual Stacked LSTM\n\n⚡ **Backend:** TensorFlow Core C++ (GPU Accelerated)")


# --- CHAT HISTORY (persists across re-runs) ---
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- HERO SECTION (only shown when no chat yet) ---
if not st.session_state.messages:
    st.markdown("<h1 style='text-align:center; margin-bottom:0.25rem;'>🤖 Welcome to my LSTM Model</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#888; margin-top:0; margin-bottom:1.25rem;'>A custom natural language processor built and trained from scratch.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.info('**Try prompting:**\n\n*"how are you"*')
    with col2:
        st.info('**Try prompting:**\n\n*"what is your"*')

    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)


# --- RENDER EXISTING CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
def generate_text_stream(seed_phrase, words_to_predict=12, temperature=0.3):
    """
    Takes a starter seed phrase, predicts the next word, appends it, 
    and loops to generate a full conversational line of text.
    """
    generated_text = seed_phrase
    print(f"Seed Prompt: '{seed_phrase}'")
    print(f"Generated Chat: {seed_phrase}", end=" ")

    for _ in range(words_to_predict):
        # 1. Map raw input text directly to integer tokens via your adapted layer
        tokens = vectorize_layer([generated_text]).numpy()[0]
        
        # Remove trailing/leading padding from the vectorization matrix
        clean_tokens = tokens[tokens != 0]
        
        # 2. Enforce structural sequence constraint of exactly 8 elements
        if len(clean_tokens) < 8:
            # Pre-pad with zeros if the sentence is shorter than context window
            padded_input = np.pad(clean_tokens, (8 - len(clean_tokens), 0), 'constant')
        else:
            # Truncate to take only the last 8 words if context length is exceeded
            padded_input = clean_tokens[-8:]
            
        # Reshape to fit model batch dimensions: (1, 8)
        input_tensor = np.array([padded_input])
        
        # 3. Generate predictions
        if(model_variant == "Model V1 (Pure Recurrent)"):
            predictions = model_v2.predict(input_tensor, verbose=0)[0]
        else:
            predictions = model_v3.predict(input_tensor, verbose=0)[0]

        # 4. Temperature Sampling (Prevents the model from getting stuck repeating words)
        # 0.6 is a good sweet spot for clean grammar with a bit of variety
        predictions = np.log(predictions + 1e-7) / temperature
        exp_preds = np.exp(predictions)
        predicted_probs = exp_preds / np.sum(exp_preds)
        
        # Sample the next token index based on the calculated probability distribution
        predicted_idx = np.random.choice(len(predicted_probs), p=predicted_probs)
        
        # 5. Reverse map the index back to a string word
        vocabulary = vectorize_layer.get_vocabulary()
        predicted_word = vocabulary[predicted_idx]
        
        # Break execution if blanks or structural flags are generated
        if predicted_word in ["", "[UNK]"]:
            continue
            
        # Append word to history for the next iteration loop pass
        generated_text += " " + predicted_word
        
    return generated_text


            

# --- CHAT INPUT ---

user_prompt = st.chat_input("Ask me anything...")

if user_prompt:
    # Save & display user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)
        

    # Generate & display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Model running tensor calculation..."):
            time.sleep(1.5)  
            
            
            response = (
                            generate_text_stream(
                    seed_phrase=user_prompt, 
                    words_to_predict=words_to_predict, 
                    temperature=temperature
                )
                )
                        

        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})