from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

# Ensure backend imports work when running: streamlit run frontend/app.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.services.medicine_service import MedicineAssistantService


st.set_page_config(page_title="Hospital AI Medicine Assistant", page_icon="🏥", layout="centered")
st.title("Hospital AI Medicine Assistant")
st.caption("Hybrid RAG: semantic medicine knowledge + structured stock lookup")

if "service" not in st.session_state or not hasattr(st.session_state.service, "update_stock"):
    st.session_state.service = MedicineAssistantService()

query = st.text_input(
    "Ask a medicine question",
    placeholder="Example: Do we have Paracetamol 500 mg in stock?",
)

if st.button("Ask") and query.strip():
    with st.spinner("Processing query..."):
        result = st.session_state.service.ask(query.strip())

    st.subheader("Answer")
    st.write(result["answer"])

    with st.expander("Debug details"):
        st.json(result)

st.divider()
st.subheader("Stock Update")
col1, col2, col3 = st.columns(3)
with col1:
    stock_med_name = st.text_input("Medicine name", key="stock_med_name")
with col2:
    stock_strength = st.text_input("Strength", key="stock_strength", placeholder="500 mg")
with col3:
    stock_delta = st.number_input("Change (+/-)", key="stock_delta", step=1, value=0)

if st.button("Update Stock"):
    if not stock_med_name.strip():
        st.error("Enter medicine name.")
    else:
        update_result = st.session_state.service.update_stock(
            medicine_name=stock_med_name.strip(),
            strength=stock_strength.strip(),
            delta=int(stock_delta),
        )
        if update_result.get("ok"):
            st.success(update_result["message"])
        else:
            st.error(update_result.get("message", "Stock update failed."))
        with st.expander("Stock update details"):
            st.json(update_result)
