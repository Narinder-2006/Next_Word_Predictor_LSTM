# 🤖 Dual-Variant LSTM Language Model Chat Interface

A premium, Gemini-inspired minimalist conversational UI for next-word prediction, powered by twin **Long Short-Term Memory (LSTM)** neural networks running natively on **TensorFlow Core C++**. 

This application tokenizes raw text inputs, formats them into structural evaluation tensors, and sequences natural language next-word completions dynamically using customizable generation hyperparameters.

---

## ⚙️ Key Architectural Features

* **Dual-Weights Selector:** Seamlessly switch between **Model V1 (Pure Recurrent)** and **Model V2 (Padded & Upgraded)** directly from the sidebar.
* **Hyperparameter Deck:** Real-time interactive controllers for generation length thresholds and creativity temperature scalers (mapping selection ranges from deterministic `0.1` up to random/creative `1.5`).
* **Optimized Hardware Pipeline:** Lazy-loads both heavy `.keras` graph structures and vocabulary JSON configuration files exactly once at boot time using `@st.cache_resource` memory mapping to completely eliminate rendering performance lag.
* **GPU Track Acceleration:** Direct bindings routing tensor calculations straight through local Nvidia CUDA allocations (`/GPU:0`) on an RTX 4060 laptop GPU setup.
* **Premium Interface Layout:** Clean, wide, left-aligned layout with customized CSS overrides to minimize container whitespace margins and lift conversational modules upward.

---

## 📁 Project Directory Map

```text
├── models/
│   ├── daily_dialog_lstm_v1.keras  # Model Variant 1 weights
│   └── daily_dialog_lstm_v2.keras  # Model Variant 2 weights
├── app.py                         # Streamlit frontend engine script
├── vocab_config.json              # TextVectorization saved vocabulary
├── requirements.txt               # Pinned environment specifications
└── .gitignore                     # Git tracking exclusions
🚀 Step-by-Step Local Deployment Guide
1. Isolate the Virtual Environment
Open your terminal inside the project root folder and execute the creation sequence:

Bash
# Create the environment
python -m venv lstm_env

# Activate on Windows (Command Prompt)
lstm_env\Scripts\activate

# Activate on Windows (PowerShell)
.\lstm_env\Scripts\activate

# Activate on Mac/Linux/Git Bash
source lstm_env/bin/activate
2. Install Pinned Dependencies
Ensure your terminal prompt displays the active (lstm_env) tag, then run the precision installation step:

Bash
pip install -r requirements.txt
3. Launch the Application Core
To completely bypass global Windows binary paths and force execution directly inside your environment's memory spaces, initialize the server using the absolute interpreter file path:

Bash
.\lstm_env\Scripts\python.exe -m streamlit run app.py
⚡ Framework Configuration Settings
Backend Runtime: TensorFlow 2.12.0

Matrix Utilities: NumPy 1.23.5

UI Interface Deck: Streamlit 1.32.0
