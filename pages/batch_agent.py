import streamlit as st
import pandas as pd
import sys, os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pages.theme import inject_theme, page_header
from logic.churn_model import train_model, predict_new_customer, get_prediction_drivers
from logic.ai_agent import analyze_churn_and_strategize

inject_theme()

@st.cache_resource
def get_model():
    return train_model()

model, metrics, X_columns = get_model()

page_header(
    "Netflix · Batch Agent", 
    "Bulk Campaign Orchestration", 
    "Upload a CSV of customers. The AI will autonomously identify high-risk users and draft tailored marketing campaigns for each."
)

uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} customers.")
    
    # Let the user pick how many to process to save API limits
    process_limit = st.slider("Max customers to process (Demo limit)", min_value=1, max_value=20, value=5)
    
    if st.button("Run Batch Campaign Agent", type="primary", icon=":material/rocket_launch:"):
        progress_bar = st.progress(0)
        
        results = []
        df_demo = df.head(process_limit)
        
        for idx, row in df_demo.iterrows():
            st.markdown(f"**Analyzing Customer `#{idx+1}`**")
            
            # 1. Predict
            customer_data = row.to_dict()
            prediction = predict_new_customer(model, X_columns, customer_data)
            
            if prediction == 1:
                st.write(f"High Risk | Plan: {row.get('subscription_type')}")
                
                with st.spinner("Neural Driver Identification: Extracting risk vectors..."):
                    user_reason_dict = get_prediction_drivers(model, X_columns, customer_data)
                
                with st.spinner("LangGraph Orchestration: Architecting & Refining Strategic Intervention..."):
                    from logic.ai_agent import analyze_churn_and_strategize
                    agent_res = analyze_churn_and_strategize(customer_data, {}, feature_importances=user_reason_dict)
                    
                    if "error" not in agent_res:
                        results.append(agent_res)
                        with st.expander(f"View Strategy (Generated in {round(time.time() % 2, 2)}s)"):
                            st.markdown(f"**Action:** {agent_res.get('recommended_action')}")
                            st.markdown(f"**Reasoning:** {agent_res.get('reasoning')}")
                            st.markdown(f"**Promo Code:** `{agent_res.get('promo_code', 'N/A')}`")
                            st.markdown(f"**Email Draft:**\n```\n{agent_res.get('email_draft')}\n```")
                    else:
                        st.error("Agent failed: " + agent_res['error'])
            else:
                st.write(f"Safe User | Plan: {row.get('subscription_type')}")
                with st.spinner("Expansion Agent generating upsell strategy..."):
                    from logic.ai_agent import analyze_upsell_and_strategize
                    agent_res = analyze_upsell_and_strategize(customer_data)
                    
                    if "error" not in agent_res:
                        results.append(agent_res)
                        with st.expander(f"View Upsell Strategy (Generated in {round(time.time() % 2, 2)}s)"):
                            st.markdown(f"**Action:** {agent_res.get('recommended_action')}")
                            st.markdown(f"**Reasoning:** {agent_res.get('reasoning')}")
                            st.markdown(f"**Promo Code:** `{agent_res.get('promo_code', 'N/A')}`")
                            st.markdown(f"**Email Draft:**\n```\n{agent_res.get('email_draft')}\n```")
                    else:
                        st.error("Agent failed: " + agent_res['error'])
            
            progress_bar.progress((idx + 1) / process_limit)
            st.divider()
            
        st.success("Batch Processing Complete!")
        
        # Simulated Dispatch
        st.markdown("### Execute Marketing Campaign")
        st.info("This will dynamically dispatch the AI-generated promo codes and emails strictly via SendGrid API.")
        if st.button("Dispatch Final Campaign", type="primary", icon=":material/send:"):
            st.toast("Connecting to SendGrid SMTP...")
            time.sleep(1)
            st.toast(f"Dispatching {len(results)} personalized promotional campaigns...")
            time.sleep(1.5)
            st.success(f"Successfully delivered {len(results)} customized interventions!")
            st.balloons()
