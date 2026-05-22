# 🤖 Financial Advisory Chatbot with Gradio & TinyLlama

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/gradio-latest-green.svg)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A sophisticated web-based financial advisory chatbot built with **Gradio** and powered by **TinyLlama** language model. This application provides real-time financial market data and AI-powered investment guidance through a beautiful, dark-themed web interface.

---

## 📋 Table of Contents

- [Features](#features)
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Support](#support)

---

## ✨ Features

- 🤖 **AI-Powered Chat** - Real-time responses using TinyLlama
- 📊 **Live Market Data** - Real-time stock prices with yfinance
- 🎨 **Beautiful UI** - Dark-themed Gradio interface
- 💼 **Financial Expertise** - Multi-stage conversation flow
- 🔄 **Multi-turn Dialogue** - Maintains conversation context
- 🚀 **Fast Performance** - Optimized for quick responses
- 📱 **Responsive Design** - Works on desktop and mobile

---

## 📖 Overview

### What Does This Do?

This Financial Advisory Chatbot:
1. Greets users with a professional interface
2. Assesses risk profile through interactive questions
3. Fetches live market data (stocks, crypto, etc.)
4. Provides AI-powered investment recommendations
5. Maintains conversation context across multiple turns
6. Delivers personalized financial advice

### Use Cases

✅ Individual investors seeking personalized advice  
✅ Portfolio management and analysis  
✅ Real-time market research  
✅ Financial education  
✅ AI capability demonstration  

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| LLM Model | TinyLlama | 1.1B parameters |
| Frontend | Gradio | 4.0+ |
| ML Library | Transformers | 4.30+ |
| Market Data | yfinance | 0.2+ |
| Deep Learning | PyTorch | 2.0+ |

### Requirements
```
Python 3.8+
├── torch (2.0.0)
├── transformers (4.30.0)
├── gradio (4.0.0)
├── yfinance (0.2.28)
├── huggingface-hub (0.16.0)
├── sentencepiece (0.1.99)
└── accelerate (0.20.0)
```

---

## 🚀 Installation

### Local Setup

#### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/financial-advisor-chatbot.git
cd financial-advisor-chatbot
```

#### Step 2: Create Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Setup Hugging Face Token
```bash
# Get token from: https://huggingface.co/settings/tokens
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### Google Colab Setup

```python
!pip install transformers gradio yfinance torch accelerate huggingface-hub sentencepiece

from google.colab import userdata
from huggingface_hub import login

token = userdata.get('HF_TOKEN')
login(token=token)
```

---

## 📖 Usage

### Running Locally

```bash
# Activate environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Run application
python app.py
```

Access at: `http://localhost:7860`

### Running in Google Colab

1. Open the Jupyter notebook
2. Add HF_TOKEN to 🔑 Secrets
3. Run all cells
4. Click the public Gradio link

### Example Interactions

```
User: "I have $10,000 and low risk tolerance"
Bot: "Based on your profile, I recommend... [live market data]"

User: "What about Tesla stock?"
Bot: "TSLA is trading at $150.25... [recommendation]"
```

---

## 🚀 Deployment

### Option 1: Gradio Public URL (Instant)

```python
demo.launch(share=True)
# Returns: https://xxxxx-xxxxx-xxxxx.gradio.live
```

**Pros:** Instant, no setup  
**Cons:** URL changes, expires in 72 hours

### Option 2: Render Hosting (Recommended) ⭐

1. **Prepare code:** Create `app.py` and `requirements.txt`
2. **Push to GitHub:** `git push origin main`
3. **Deploy on Render:**
   - Go to https://render.com
   - Click "New Web Service"
   - Select GitHub repo
   - Set build: `pip install -r requirements.txt`
   - Set start: `python app.py`
   - Add environment: `HF_TOKEN=[your-token]`
4. **Share link:** `https://financial-advisor.onrender.com`

**Cost:** FREE | **Time:** 15 minutes | **Duration:** Permanent

### Option 3: Vercel + Backend

See detailed guide in [DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### "HF_TOKEN not found"
- Add to environment variables
- In Colab: Click 🔑 Secrets, add HF_TOKEN
- In Render: Settings → Environment Variables

### "Port already in use"
```bash
# Use different port
python app.py --server_port 7861
```

### "Model not found"
```python
from huggingface_hub import model_info
info = model_info("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more help.

---

## 📁 Project Structure

```
financial-advisor-chatbot/
├── README.md
├── requirements.txt
├── .env.example
├── gradio_frontend.ipynb          # Main notebook
├── app.py                          # Production app
├── config.py                       # Configuration
├── utils.py                        # Utilities
├── docs/
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── TROUBLESHOOTING.md
└── deployment/
    ├── Dockerfile
    └── render.yaml
```

---

## 📊 Project Statistics

```
Language: Python
Total Lines: 500+
Models: TinyLlama (extensible)
APIs: 3+
Last Updated: May 2026
Status: ✅ Active
```

---

## 🗺️ Roadmap

### v1.0 (Current)
- ✅ Chat interface
- ✅ Market data
- ✅ Risk assessment
- ✅ Gradio UI

### v1.1 (Planned)
- 🔄 Multi-language
- 🔄 Portfolio analytics
- 🔄 Investment history

### v2.0 (Future)
- 🔮 Real-time collaboration
- 🔮 Mobile app
- 🔮 Advanced models

---

## 🙋 FAQ

**Q: Can I use this commercially?**
A: Yes, under MIT license. Ensure financial compliance.

**Q: Is my data stored?**
A: No, conversations are not stored by default.

**Q: Can I use different models?**
A: Yes, swap `TinyLlama` with any Hugging Face model.

**Q: How accurate is the advice?**
A: This is AI, not a certified advisor. Verify with professionals.

**Q: How much does it cost?**
A: FREE on Render or Colab.

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes and commit: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📞 Support

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Email:** your-email@example.com

**Resources:**
- [Gradio Docs](https://gradio.app/docs)
- [Hugging Face Hub](https://huggingface.co/docs)
- [PyTorch Docs](https://pytorch.org/docs)

---

## 🙏 Acknowledgments

- **TinyLlama Team** - Excellent language model
- **Gradio** - Beautiful UI framework
- **Hugging Face** - Model hosting
- **yfinance** - Market data
- **Contributors** - Community support

---

## ⭐ Show Your Support

If this project helped you, please give it a star on GitHub!

---

<div align="center">

**Made with ❤️ by Your Sharon Manohar

[⬆ Back to Top](#-financial-advisory-chatbot-with-gradio--tinyllama)

</div>

---

**Version:** 1.0  
**Last Updated:** May 22, 2026  
**Status:** ✅ Active & Maintained  
**License:** MIT

