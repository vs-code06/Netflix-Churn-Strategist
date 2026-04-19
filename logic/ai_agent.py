import os
import json
from typing import TypedDict, Annotated, List, Union
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from logic.rag_system import get_relevant_strategies
import streamlit as st

# --- State Definition ---
class AgentState(TypedDict):
    customer_profile: dict
    churn_reason: str
    feature_importances: dict
    rag_context: str
    feedback_context: str
    strategy: dict
    critic_feedback: str
    iterations: int
    improvement_needed: bool
    mode: str # 'retention' or 'expansion'
    error: str

# --- Helper Functions ---
def get_groq_api_key():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("GROQ_API_KEY")
    if key: return key
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key: return key
    except Exception: pass
    return None

def get_historical_feedback():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(ROOT_DIR, "data", "agent_feedback_log.json")
    if not os.path.exists(log_path): return "No historical feedback available."
    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
            if not logs: return "No historical feedback available."
            feedback_str = "Critical Previous Campaign Feedback:\n"
            for entry in logs[-5:]:
                feedback_str += f"- Profile: {entry.get('plan')} | Action: {entry.get('action')} | Outcome: {entry.get('status')} | Feedback: {entry.get('feedback')}\n"
            return feedback_str
    except Exception: return "No historical feedback available."

# --- Graph Nodes ---

def researcher_node(state: AgentState):
    """Gathers RAG context and historical feedback."""
    profile_str = ", ".join([f"{k}: {v}" for k, v in state['customer_profile'].items()])
    rag_context = get_relevant_strategies(profile_str, state['churn_reason'])
    feedback_context = get_historical_feedback()
    return {
        "rag_context": rag_context, 
        "feedback_context": feedback_context,
        "iterations": 0
    }

def architect_node(state: AgentState):
    """Generates or refines the strategic JSON with elite marketing psychology."""
    api_key = get_groq_api_key()
    if not api_key: return {"error": "GROQ_API_KEY not found."}
    
    llm = ChatGroq(
        temperature=0.75,
        model_name="llama-3.3-70b-versatile",
        api_key=api_key,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    system_prompt = (
        "You are the Global Chief Marketing Officer at Netflix. You output ONLY valid JSON.\n"
        "Your mission is to generate 'Elite Tier' B2C communications using the AIDA Framework (Attention, Interest, Desire, Action).\n\n"
        "FORBIDDEN CLICHES (NEVER USE):\n"
        "- 'We missed you', 'Please come back', 'Sorry to see you go', 'We want you back', 'Don't leave us'.\n\n"
        "TONE GUIDELINES:\n"
        "- Use sophisticated, value-driven, and exclusive language.\n"
        "- Focus on content rarity and personalized experience re-evaluation.\n"
        "- Subject lines must be intriguing and prestigious (e.g., 'An Exclusive Perspective on Your Premium Membership').\n\n"
        "GOLD STANDARD EXAMPLE:\n"
        "{{\n"
        "  'reasoning': 'The user exhibits high LTV potential but has hit a content discovery plateau. Engagement depth has dropped 40% in 15 days.',\n"
        "  'recommended_action': '- Deploy 48-hour Early Access to upcoming AAA titles.\\n- Personalize Genre Dashboard for Sci-Fi optimization.',\n"
        "  'email_draft': 'Subject: Elevating Your Entertainment Standards\\nDear XYZ,\\nAt Netflix, cinematic excellence is a moving target. We have curated a bespoke preview of this month\\'s most exclusive premieres specifically aligned with your preference for Sci-Fi narrative depth...',\n"
        "  'promo_code': 'PRESTAGE-XYZ-2026'\n"
        "}}\n\n"
        "Your response MUST be a single flat JSON object with these EXACT keys:\n"
        "1. 'reasoning': string (2 sentences)\n"
        "2. 'recommended_action': string (markdown bulleted list)\n"
        "3. 'email_draft': string (Start with 'Subject: ...' followed by the body)\n"
        "4. 'promo_code': string"
    )
    
    if state['mode'] == 'retention':
        human_content = (
            f"STRATEGIC TASK: Retention Intervention for Churn Risk User.\n"
            f"Customer Profile: {state['customer_profile']}\n"
            f"Primary Risk Drivers: {state['churn_reason']}\n"
            f"Knowledge Base Strategies: {state['rag_context']}\n"
            f"Historical Feedback Logs: {state['feedback_context']}\n"
        )
    else:
        human_content = (
            f"STRATEGIC TASK: LTV Expansion for Engaged User.\n"
            f"Customer Profile: {state['customer_profile']}\n"
            f"Context: {state['rag_context']}\n"
        )

    if state.get('critic_feedback'):
        human_content += f"\nCRITICAL CMO FEEDBACK (REJECTED DRAFT): {state['critic_feedback']}\nRefine the above into a 'Prestige' version addressing these exact flaws."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), 
        ("human", "{content}")
    ])
    chain = prompt | llm
    
    try:
        response = chain.invoke({"content": human_content})
        strategy = json.loads(response.content)
        return {"strategy": strategy, "iterations": state['iterations'] + 1, "critic_feedback": ""}
    except Exception as e:
        return {"error": f"Architect Error: {str(e)}"}

def critic_node(state: AgentState):
    """Reviews the draft for 'Prestige' marketing quality and compliance."""
    if state.get('error'): return state
    
    strategy = state.get('strategy', {})
    
    # 1. Type Check
    email_draft = strategy.get('email_draft', '')
    if isinstance(email_draft, dict):
        email_draft = " ".join([f"{k}: {v}" for k, v in email_draft.items()])
    
    email_text = str(email_draft).lower()
    email_subject = str(email_draft).split('\n')[0].lower()
    
    # 2. Advanced Validation Logic
    issues = []
    
    # Check for Forbidden Cliches
    forbidden = ["missed you", "come back", "sorry to see", "don't leave", "want you back"]
    if any(phrase in email_text for phrase in forbidden):
        issues.append("Email contains banned generic clichés. Refine for elite, value-driven tone.")
    
    # Check for Length/Sophistication
    if len(str(strategy.get('email_draft', ''))) < 200:
        issues.append("Email draft is too brief/shallow. Expand on value propositions and narrative depth.")
    
    if state['mode'] == 'retention' and state['churn_reason'].lower() not in str(strategy.get('reasoning', '')).lower():
        issues.append(f"Strategic reasoning ignores the primary churn driver: {state['churn_reason']}.")
    
    if issues:
        return {"improvement_needed": True, "critic_feedback": " | ".join(issues)}
    return {"improvement_needed": False, "critic_feedback": ""}

# --- Graph Construction ---

def should_continue(state: AgentState):
    if state.get('error') or state['iterations'] >= 2:
        return "end"
    if state.get('improvement_needed'):
        return "refine"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("architect", architect_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "architect")
workflow.add_edge("architect", "critic")
workflow.add_conditional_edges("critic", should_continue, {"refine": "architect", "end": END})

app = workflow.compile()

# --- Public API ---

def analyze_churn_and_strategize(customer_profile: dict, prediction_metrics: dict, feature_importances: dict = None):
    churn_reason = "General Friction"
    if feature_importances:
        top_features = list(feature_importances.keys())[:3]
        churn_reason = f"Top contributing factors: {', '.join(top_features)}"
    
    initial_state = {
        "customer_profile": customer_profile,
        "churn_reason": churn_reason,
        "feature_importances": feature_importances,
        "mode": "retention",
        "iterations": 0,
        "critic_feedback": "",
        "improvement_needed": False
    }
    
    try:
        final_state = app.invoke(initial_state)
        if final_state.get('error'): return {"error": final_state['error']}
        return final_state['strategy']
    except Exception as e:
        return {"error": f"Graph Execution Error: {str(e)}"}

def analyze_upsell_and_strategize(customer_profile: dict):
    initial_state = {
        "customer_profile": customer_profile,
        "churn_reason": "Highly engaged expansion opportunity",
        "feature_importances": {},
        "mode": "expansion",
        "iterations": 0,
        "critic_feedback": "",
        "improvement_needed": False
    }
    try:
        final_state = app.invoke(initial_state)
        if final_state.get('error'): return {"error": final_state['error']}
        return final_state['strategy']
    except Exception as e:
        return {"error": f"Graph Execution Error: {str(e)}"}

if __name__ == "__main__":
    test_profile = {"age": 30, "subscription_type": "Basic", "watch_hours": 2.0}
    print("Testing LangGraph Workflow...")
    res = analyze_churn_and_strategize(test_profile, {}, {"watch_hours": 0.8})
    print(json.dumps(res, indent=2))
