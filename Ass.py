import sys
!{sys.executable} -m pip install streamlit
!{sys.executable} -m pip install transformers
import streamlit as st
from transformers import pipeline

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="International Business Marketing Prompt App",
    page_icon="🌍",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    generator = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M-Instruct"
    )
    return generator

generator = load_model()

# -----------------------------
# App Title
# -----------------------------
st.title("🌍 International Business Marketing Prompt Application")

st.markdown("""
This AI application generates:

- ✅ Global-Ready Product Title  
- ✅ Powerful Marketing Slogan  
- ✅ Advertising Description from 3 Expert Perspectives  

using Generative AI and Prompt Engineering.
""")

# -----------------------------
# User Input
# -----------------------------
product_name = st.text_input(
    "Enter Product Name",
    placeholder="Example: Smart Water Bottle"
)

generate_btn = st.button("Generate Marketing Content")

# -----------------------------
# Prompt Template
# -----------------------------
def create_prompt(product):
    return f"""
You are an international business marketing expert.

Generate professional global marketing content for the product: "{product}"

Requirements:
1. Create a global-ready product title.
2. Create a powerful emotional marketing slogan.
3. Write advertising descriptions from THREE expert perspectives:
   - Luxury Brand Expert
   - Digital Marketing Expert
   - Emotional Storytelling Expert

The response should:
- Follow international marketing standards
- Be persuasive and professional
- Appeal to global audiences
- Use emotional engagement strategies
- Be suitable for advertising campaigns

Format:

Global Product Title:
...

Marketing Slogan:
...

1. Luxury Brand Expert:
...

2. Digital Marketing Expert:
...

3. Emotional Storytelling Expert:
...
"""

# -----------------------------
# Generate Content
# -----------------------------
if generate_btn:

    if product_name.strip() == "":
        st.warning("Please enter a product name.")
    else:

        prompt = create_prompt(product_name)

        with st.spinner("Generating global marketing content..."):

            result = generator(
                prompt,
                max_new_tokens=400,
                temperature=0.8,
                do_sample=True
            )

            output = result[0]["generated_text"]

        st.success("Marketing Content Generated Successfully!")

        st.markdown("## ✨ Generated Content")
        st.write(output)
