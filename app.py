#!/usr/bin/env python3
"""
Financial Advisory Chatbot - Complete Working Version
Converted from Jupyter Notebook to standalone Python script
"""

import os
import torch
import gradio as gr
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

print("=" * 60)
print("Financial Advisory Chatbot - Starting")
print("=" * 60)

# ============================================================================
# SETUP & AUTHENTICATION
# ============================================================================

print("\n[1/5] Setting up authentication...")
try:
    token = os.environ.get('HF_TOKEN')
    if token:
        login(token=token)
        print("✓ HuggingFace authentication successful")
    else:
        print("⚠ Warning: HF_TOKEN not set")
except Exception as e:
    print(f"⚠ Authentication warning: {e}")

# ============================================================================
# SYSTEM INFO
# ============================================================================

print(f"✓ GPU Available: {torch.cuda.is_available()}")
print(f"✓ Gradio Version: {gr.__version__}")
print("✓ All imports successful!")

# ============================================================================
# CLEAR MEMORY & MODEL LOADING
# ============================================================================

print("\n[2/5] Loading model...")
torch.cuda.empty_cache()

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("    Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("    Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

model.eval()
print("✓ Model loaded successfully!")
try:
    print(f"✓ Memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
except:
    pass

# ============================================================================
# RESPONSE GENERATION FUNCTION
# ============================================================================

print("\n[3/5] Setting up generation function...")

def generate_response(conversation_history, max_new_tokens=300):
    """Generate response using the model"""
    # Build prompt
    prompt = ""
    
    for message in conversation_history:
        if message["role"] == "system":
            prompt += f"<|system|>\n{message['content']}</s>\n"
        elif message["role"] == "user":
            prompt += f"<|user|>\n{message['content']}</s>\n"
        elif message["role"] == "assistant":
            prompt += f"<|assistant|>\n{message['content']}</s>\n"
    
    prompt += "<|assistant|>\n"
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode
    full_response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    
    # Extract only assistant response
    if "<|assistant|>" in full_response:
        response = full_response.split("<|assistant|>")[-1].strip()
    else:
        response = full_response.strip()
    
    return response

print("✓ Generation function ready!")

# ============================================================================
# CONFIGURATION & GLOBAL VARIABLES
# ============================================================================

print("\n[4/5] Initializing configuration...")

# Define assistant name
ASSISTANT_NAME = "FinBot"

QUESTIONS = {
    1: f"Hello! 👋 I am {ASSISTANT_NAME}, your Personal Financial Advisor. What is your full name?",
    2: "Nice to meet you, {name}! How old are you and what is your current occupation?",
    3: "Thank you! What is your monthly income in dollars?",
    4: "Got it! What are your total monthly expenses in dollars?",
    5: "Noted! How much do you currently have in savings?",
    6: "Great! What is your main financial goal? For example: buying a house, retirement, or emergency fund?",
    7: "Almost done! What is your risk tolerance? Choose: Conservative, Moderate, or Aggressive?",
    8: "Thank you {name}! Generating your personalized financial plan now... 🔄"
}

# Global variables
conversation_history = []
current_stage = 1
user_data = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# MARKET DATA FUNCTION
# ============================================================================

def get_market_data():
    """Fetch real market data"""
    tickers = {
        "S&P 500 ETF": "SPY",
        "Gold ETF": "GLD",
        "Bond ETF": "BND",
        "Tech ETF": "QQQ"
    }
    
    market_info = []
    for name, ticker in tickers.items():
        try:
            stock = yf.Ticker(ticker)
            info = stock.history(period="1d")
            if not info.empty:
                price = info['Close'].iloc[-1]
                market_info.append(f"{name} ({ticker}): ${price:.2f}")
        except:
            market_info.append(f"{name}: Data unavailable")
    
    return "\n".join(market_info)

# Test market data
print("    Testing market data...")
try:
    market_data = get_market_data()
    print(f"✓ Market data retrieved:\n{market_data}")
except Exception as e:
    print(f"⚠ Market data warning: {e}")

# ============================================================================
# CHAT LOGIC & MANAGEMENT
# ============================================================================

def get_progress_text():
    """Get progress text"""
    global current_stage
    stages = [
        "👋 Greeting",
        "👤 Collecting Name",
        "🎂 Age & Occupation",
        "💵 Monthly Income",
        "💳 Monthly Expenses",
        "💰 Current Savings",
        "🎯 Financial Goals",
        "⚖️ Risk Tolerance",
        "📋 Generating Plan"
    ]
    if current_stage <= len(stages):
        return f"Stage {current_stage}/{len(stages)}: {stages[current_stage-1]}"
    return "✅ Financial plan complete!"

def initialize_conversation():
    """Initialize conversation"""
    global conversation_history, current_stage, user_data
    
    conversation_history = []
    current_stage = 1
    user_data = {}
    
    opening = QUESTIONS[1]
    print(f"Opening message: {opening}")
    return opening

def reset_conversation():
    """Reset conversation"""
    global conversation_history, current_stage, user_data
    
    opening = initialize_conversation()
    return [[None, opening]]

def generate_final_recommendation():
    """Generate final recommendation using LLM"""
    try:
        market_data = get_market_data()
    except:
        market_data = "Market data unavailable"
    
    final_prompt = f"""<|system|>
You are {ASSISTANT_NAME}, an expert financial advisor.
Give clear specific financial advice.</s>
<|user|>
Create a personalized financial plan for this client:

Name: {user_data.get('name', 'Client')}
Age and Job: {user_data.get('age_occupation', 'Not provided')}
Monthly Income: {user_data.get('income', 'Not provided')}
Monthly Expenses: {user_data.get('expenses', 'Not provided')}
Current Savings: {user_data.get('savings', 'Not provided')}
Financial Goal: {user_data.get('goals', 'Not provided')}
Risk Tolerance: {user_data.get('risk', 'Not provided')}

Live Market Data:
{market_data}

Structure your response exactly like this:

💰 FINANCIAL SITUATION SUMMARY
Write 2 sentences about their situation.

🎯 GOAL ANALYSIS
Comment on their specific goals.

📈 INVESTMENT STRATEGY
Give 3 specific recommendations based on risk level.

✅ 5 ACTION STEPS
List 5 practical steps they can take this week.

💡 CLOSING MESSAGE
One encouraging sentence to end.</s>
<|assistant|>
"""
    
    inputs = tokenizer(
        final_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    return response

def chat(message, history):
    """Main chat function"""
    global current_stage, conversation_history, user_data
    
    if not message:
        return "", history
    
    conversation_history.append({"role": "user", "content": message})
    
    # Stage-specific logic
    if current_stage == 1:
        user_data['name'] = message
        current_stage = 2
        next_question = QUESTIONS[2]
    elif current_stage == 2:
        user_data['age_occupation'] = message
        current_stage = 3
        next_question = QUESTIONS[3]
    elif current_stage == 3:
        user_data['income'] = message
        current_stage = 4
        next_question = QUESTIONS[4]
    elif current_stage == 4:
        user_data['expenses'] = message
        current_stage = 5
        next_question = QUESTIONS[5]
    elif current_stage == 5:
        user_data['savings'] = message
        current_stage = 6
        next_question = QUESTIONS[6]
    elif current_stage == 6:
        user_data['goals'] = message
        current_stage = 7
        next_question = QUESTIONS[7]
    elif current_stage == 7:
        user_data['risk'] = message
        current_stage = 8
        next_question = QUESTIONS[8]
        next_question = next_question.format(name=user_data.get('name', 'Client'))
    elif current_stage >= 8:
        next_question = generate_final_recommendation()
    else:
        next_question = "How can I help?"
    
    conversation_history.append({"role": "assistant", "content": next_question})
    history.append((message, next_question))
    
    return "", history

print("✓ Chat functions ready!")

# ============================================================================
# BUILD GRADIO INTERFACE
# ============================================================================

print("\n[5/5] Building Gradio interface...")

with gr.Blocks(
    title="Financial Advisory Chatbot",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="gray"
    ),
    css="""
    body { background: linear-gradient(135deg, #0f0f1e, #1a1a2e); }
    .gradio-container { background: transparent; }
    #chatbot { background: #16213e; border: 1px solid #0f3460; }
    #msg_input { background: #16213e; border: 1px solid #0f3460; color: white; }
    .progress-box { background: #16213e; border: 1px solid #0f3460; }
    .info-box { background: #16213e; border: 1px solid #0f3460; }
    """
) as demo:
    
    # Header
    gr.Markdown("""
    <div style="text-align:center; padding:20px;">
        <h1 style="color:#00d4ff;">💰 Financial Advisory Chatbot</h1>
        <p style="color:#aaa;">Powered by TinyLlama • Real-time Market Data • AI Recommendations</p>
    </div>
    """)
    
    with gr.Tabs():
        # TAB 1: MAIN CHAT
        with gr.Tab("💬 Chat"):
            
            # Stats boxes
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="
                        background: linear-gradient(135deg, #1a1a2e, #16213e);
                        border: 1px solid rgba(0,212,255,0.3);
                        border-radius: 15px;
                        padding: 15px;
                        text-align: center;
                    ">
                        <div style="font-size:2em;">🤖</div>
                        <div style="color:#00d4ff; font-weight:bold; font-size:1.1em;">
                            AI Powered
                        </div>
                        <div style="color:white; font-size:0.9em;">
                            TinyLlama Model
                        </div>
                    </div>
                    """)
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="
                        background: linear-gradient(135deg, #1a1a2e, #16213e);
                        border: 1px solid rgba(247,151,30,0.3);
                        border-radius: 15px;
                        padding: 15px;
                        text-align: center;
                    ">
                        <div style="font-size:2em;">📈</div>
                        <div style="color:#ffd200; font-weight:bold; font-size:1.1em;">
                            Live Market Data
                        </div>
                        <div style="color:white; font-size:0.9em;">
                            Real Time Prices
                        </div>
                    </div>
                    """)
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="
                        background: linear-gradient(135deg, #1a1a2e, #16213e);
                        border: 1px solid rgba(255,100,100,0.3);
                        border-radius: 15px;
                        padding: 15px;
                        text-align: center;
                    ">
                        <div style="font-size:2em;">🎯</div>
                        <div style="color:#ff6b6b; font-weight:bold; font-size:1.1em;">
                            Personalized
                        </div>
                        <div style="color:white; font-size:0.9em;">
                            Custom Strategy
                        </div>
                    </div>
                    """)
            
            # Progress bar
            with gr.Row():
                progress_box = gr.Textbox(
                    label="📊 Conversation Progress",
                    value="👋 Click 'Start New Conversation' to begin!",
                    interactive=False,
                    elem_classes=["progress-box"]
                )
            
            # Chatbot
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="💬 Chat with your Financial Advisor",
                bubble_full_width=False,
                height=450
            )
            
            # Input area
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="💬 Type your response here...",
                    label="Your Message",
                    scale=5,
                    elem_id="msg_input"
                )
                send_btn = gr.Button(
                    "Send 📤",
                    variant="primary",
                    scale=1,
                    elem_id="send_btn"
                )
            
            # Action buttons
            with gr.Row():
                start_btn = gr.Button(
                    "🚀 Start New Conversation",
                    variant="secondary",
                    elem_id="start_btn"
                )
                refresh_btn = gr.Button(
                    "🔄 Refresh Market Data",
                    variant="secondary",
                    elem_id="refresh_btn"
                )
        
        # TAB 2: MARKET DATA
        with gr.Tab("📈 Live Market Data"):
            gr.Markdown("### 📊 Real-Time Market Overview")
            
            with gr.Row():
                market_box = gr.Textbox(
                    label="📈 Current Market Prices",
                    value=get_market_data(),
                    interactive=False,
                    lines=8,
                    elem_classes=["info-box"]
                )
            
            with gr.Row():
                gr.Markdown("""
                ### Investment Options Guide
                - **SPY** - S&P 500 ETF: Tracks top 500 US companies
                - **GLD** - Gold ETF: Safe haven asset
                - **BND** - Bond ETF: Lower risk fixed income
                - **QQQ** - Tech ETF: Top 100 Nasdaq companies
                """)
            
            refresh_market_btn = gr.Button(
                "🔄 Refresh Market Data",
                variant="primary"
            )
        
        # TAB 3: HOW IT WORKS
        with gr.Tab("ℹ️ How It Works"):
            gr.Markdown("""
            ## 🤖 How Your Financial Assistant Works
            
            **Step 1 — 👋 Greeting:** The AI introduces itself
            **Step 2 — 👤 Personal Info:** Collects your age and occupation
            **Step 3 — 💵 Financial Situation:** Asks about income, expenses and savings
            **Step 4 — 🎯 Goals & Risk:** Understands your goals and risk tolerance
            **Step 5 — 📋 AI Recommendation:** Generates personalized strategy with market data
            """)
    
    # Footer
    gr.Markdown("""
    <div style="text-align:center; padding:20px; color:#666;">
        ⚠️ Always consult a qualified financial advisor for real decisions
        <br>
        Built with 🤖 TinyLlama • 🐍 Python • 🎨 Gradio
    </div>
    """)
    
    # ============================================================================
    # EVENT HANDLERS
    # ============================================================================
    
    def handle_send(message, history):
        """Handle send button"""
        empty, updated_history = chat(message, history)
        progress = get_progress_text()
        return empty, updated_history, progress
    
    def handle_start():
        """Handle start button"""
        history = reset_conversation()
        progress = get_progress_text()
        return history, progress
    
    def handle_refresh():
        """Handle refresh button"""
        return get_market_data()
    
    # Connect events
    send_btn.click(
        handle_send,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot, progress_box]
    )
    
    msg_box.submit(
        handle_send,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot, progress_box]
    )
    
    start_btn.click(
        handle_start,
        outputs=[chatbot, progress_box]
    )
    
    refresh_btn.click(
        handle_refresh,
        outputs=[market_box]
    )
    
    refresh_market_btn.click(
        handle_refresh,
        outputs=[market_box]
    )

print("✓ Gradio interface built!")

# ============================================================================
# LAUNCH
# ============================================================================

print("\n" + "=" * 60)
print("LAUNCHING GRADIO APP")
print("=" * 60)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 7860))
    print(f"\nPort: {port}")
    print(f"Server: 0.0.0.0")
    print("\nStarting server...\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
