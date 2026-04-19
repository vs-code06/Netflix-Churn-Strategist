import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pages.theme import inject_theme, page_header, section_label
from logic.churn_model import train_model, predict_new_customer, get_prediction_drivers
from logic.ai_agent import analyze_churn_and_strategize, analyze_upsell_and_strategize
import json

inject_theme()

def save_feedback(status, feedback_text, plan, action):
    log_path = os.path.join(os.path.dirname(__file__), "..", "data", "agent_feedback_log.json")
    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
    except FileNotFoundError:
        logs = []
    logs.append({"status": status, "feedback": feedback_text, "plan": plan, "action": action})
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=4)

@st.cache_resource
def get_model():
    return train_model()

model, metrics, X_columns = get_model()

page_header(
    "Netflix · AI Strategist",
    "Agentic Churn Analysis",
    "Input a customer profile. An AI Agent will consult past marketing strategies and generate a custom retention plan."
)

with st.form("agent_form"):
    section_label("Customer Profile", margin_top="0")
    c1, c2, c3, c4 = st.columns(4)
    age    = c1.number_input("Age",    min_value=18, max_value=100, value=22)
    gender = c2.selectbox("Gender",  ['Male', 'Female', 'Other'])
    region = c3.selectbox("Region",  ['Africa', 'Europe', 'Asia', 'Oceania', 'South America', 'North America'], index=2)
    plan   = c4.selectbox("Plan",    ['Basic', 'Standard', 'Premium'])
    
    section_label("Engagement Indicators")
    c5, c6, c7, c8 = st.columns(4)
    watch_hours = c5.number_input("Monthly Watch Hrs", min_value=0.0, max_value=1000.0, value=8.5)
    last_login  = c6.number_input("Days Since Login", min_value=0, max_value=365, value=15)
    fee         = c7.number_input("Monthly Fee ($)", min_value=0.0, value=9.99, step=0.01)
    genre       = c8.selectbox("Favourite Genre", ['Action', 'Sci-Fi', 'Drama', 'Horror', 'Romance', 'Comedy', 'Documentary'])
    
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Run Agent Analysis", type="primary", use_container_width=True, icon=":material/smart_toy:")

if submitted:
    customer_data = {
        'age': age, 'gender': gender,
        'subscription_type': plan,
        'watch_hours': watch_hours,
        'last_login_days': last_login,
        'region': region, 'device': 'Mobile',
        'monthly_fee': fee,
        'payment_method': 'Credit Card',
        'number_of_profiles': 1,
        'avg_watch_time_per_day': watch_hours / 30,
        'favorite_genre': genre,
    }
    st.session_state.customer_data = customer_data
    
    with st.status("Initializing Agentic Intelligence Layer...", expanded=True) as status:
        st.write("1. **Predictive Analytics**: Initializing Scikit-Learn Decision Tree for risk categorization...")
        prediction = predict_new_customer(model, X_columns, customer_data)
        
        if prediction == 0:
            st.write("2. **Safe Signal**: Routing to Expansion Node for LTV maximization...")
            st.write("3. **Upsell Synthesis**: Orchestrating LLaMA-3 for growth-focused strategy...")
            agent_result = analyze_upsell_and_strategize(customer_data)
        else:
            st.write("2. **Neural Driver Identification**: Extracting localized behavioral risk vectors...")
            feature_imp = get_prediction_drivers(model, X_columns, customer_data)
            st.write(f"Detected Churn Vectors: `{', '.join(feature_imp.keys()) if feature_imp else 'General Friction'}`")
            
            st.write("3. **Contextual Retrieval**: Traversing Vector DB and historical RL feedback logs...")
            st.write("4. **Strategy Architecture**: Orchestrating Architect Agent for multi-objective retention drafting...")
            st.write("5. **Agentic Refinement**: Initiating Critic Agent for heuristic evaluation and iterative self-correction...")
            agent_result = analyze_churn_and_strategize(customer_data, {}, feature_importances=feature_imp)
            
        if "error" in agent_result:
            status.update(label="Agent Error", state="error")
            st.session_state.agent_result = {"error": agent_result["error"]}
        else:
            status.update(label="Agent Analysis Complete!", state="complete")
            agent_result["prediction"] = int(prediction)
            st.session_state.agent_result = agent_result

# Display Persistent UI
if "agent_result" in st.session_state:
    res = st.session_state.agent_result
    
    if "error" in res:
        st.error(res["error"])
    else:
        is_churn = res.get("prediction", 1) == 1
        st.markdown("### Agent Recommendations")
        if not is_churn:
            st.success("Target Classification: **Expansion/Upsell (Safe User)**")
        else:
            st.warning("Target Classification: **Churn Risk Recovery**")
            
        st.info(f"**Reasoning:** {res.get('reasoning', 'No reasoning provided.')}")
        
        cA, cB = st.columns(2)
        with cA:
            st.success(f"**Strategic Action:**\n\n{res.get('recommended_action', 'No action provided.')}")
            st.markdown(f"**Generated Stripe Promo Code:** `{res.get('promo_code', 'N/A')}`")
            
            st.divider()
            st.markdown("##### Reinforcement Learning Feedback")
            st.caption("Provide feedback to tune the AI mapping dynamically.")
            f1, f2 = st.columns(2)
            if f1.button("Strategy Worked", icon=":material/thumb_up:"):
                plan = st.session_state.customer_data.get('subscription_type', 'Unknown')
                save_feedback("Success", "Strategy successfully retained or upgraded the user.", plan, res.get('promo_code', 'Standard Offer'))
                st.toast("RL Feedback Logged!")
            if f2.button("Strategy Failed", icon=":material/thumb_down:"):
                plan = st.session_state.customer_data.get('subscription_type', 'Unknown')
                save_feedback("Failure", "The user rejected the offer or churned anyway.", plan, res.get('promo_code', 'Standard Offer'))
                st.toast("RL Failure Logged!")
                
        with cB:
            st.markdown("##### Drafted Email")
            st.text_area("Email Draft", value=res.get('email_draft', 'No draft provided.'), height=200, label_visibility="collapsed")
