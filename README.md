# 🌱 Calmera AI 

**Your Compassionate Wellness Companion**

Calmera AI  is an empathetic mental wellness chatbot built with LangChain and Cohere, designed to provide a safe space for thoughts and feelings through AI-powered conversations.

[Streamlit](https://calmeraaichatbot-nynkljrzdnee5hd9gcwiwq.streamlit.app/)

## ✨ Features

- 🤖 **Empathetic Conversations**: Powered by Cohere's advanced language models
- 📚 **RAG Integration**: Access evidence-based wellness guidance from uploaded PDF documents
- 🎭 **Emotional Analysis**: Dual NLP approach using TextBlob and VADER for accurate sentiment detection
- 🧠 **Emotion Detection**: Identifies specific emotions (joy, sadness, anxiety, fear, hope, etc.)
- 🚨 **Crisis Detection**: Identifies crisis situations and provides appropriate resources
- 💭 **Mood Tracking**: Summarizes emotional states throughout conversations
- 🎨 **User-Friendly Interface**: Clean, calming Streamlit interface designed for wellness

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM Framework**: LangChain
- **Language Model**: Cohere
- **Vector Store**: FAISS
- **Embeddings**: Cohere Embeddings
- **NLP & Sentiment Analysis**: 
  - TextBlob (Polarity & Subjectivity)
  - VADER Sentiment (Social media optimized)
- **PDF Processing**: PyPDF / LangChain Document Loaders

## 📋 Prerequisites

- Python 3.8 or higher
- Cohere API key ([Get one here](https://cohere.com/))

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/SamadhiJagathsiri/CalmeraAI_Chatbot
cd CalmeraAI_Chatbot
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```env
COHERE_API_KEY=your_cohere_api_key_here
```

5. **Add wellness guides (optional)**
Place PDF wellness guides in the `data/guides/` directory to enable RAG functionality.

## 🎯 Usage

1. **Start the application**
```bash
streamlit run app.py
```

2. **Access the app**
Open your browser and navigate to `http://localhost:8501`

3. **Start chatting**
Share your thoughts and feelings in the chat interface. MindEase will respond with empathy and support.

## 📁 Project Structure

```

mindease/
│
├── app.py                          ← Enhanced Streamlit UI
├── chatbot/
│   ├── __init__.py
│   ├── mindease_ai.py              ← LangChain orchestration layer
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── conversation_chain.py   ← Main conversation logic
│   │   ├── rag_chain.py            ← Retrieval for wellness guides
│   │   └── reflection_chain.py     ← Self-reflection prompts
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation_memory.py  ← Enhanced memory management
│   ├── crisis_detection.py         ← Safety & crisis rules
│   ├── sentiment_analysis.py       ← # NLP sentiment analysis (TextBlob + VADER)
│   └── prompts/
│       ├── __init__.py
│       └── templates.py            ← Prompt templates
│
├── data/
│   ├── guides/                     ← PDF wellness resources
│   └── vectorstore/                ← Chroma/FAISS embeddings
│
├── utils/
│   ├── __init__.py
│   ├── config.py                   ← Configuration
│   ├── document_loader.py          ← PDF processing
│   └── vectorstore_manager.py      ← Vector DB management
│
├── requirements.txt
└── README.md

```

## ⚙️ Configuration

### Settings Panel
- **Show emotional analysis**: Toggle sentiment display
- **Recent mood**: View emotional state summary
- **New Conversation**: Clear chat history
- **About MindEase**: Learn more about the features

### RAG (Retrieval-Augmented Generation)
To enable evidence-based responses:
1. Add PDF documents to `data/guides/`
2. Restart the application
3. The system will automatically index the documents

## 🔒 Privacy & Safety

- **No Data Storage**: Conversations are not permanently stored
- **Crisis Detection**: Automatically identifies crisis situations
- **Professional Disclaimer**: Clear messaging that this is not a replacement for professional care
- **Emergency Resources**: Provides crisis helpline information when needed

## ⚠️ Important Disclaimer

**Calmera AI is a supportive tool and NOT a substitute for professional mental health care.** If you're experiencing a mental health crisis, please:
- Contact emergency services 
- Call a crisis helpline
- Reach out to a mental health professional

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 🙏 Acknowledgments

- [Cohere](https://cohere.com/) for their powerful language models
- [LangChain](https://langchain.com/) for the RAG framework
- [Streamlit](https://streamlit.io/) for the amazing UI framework

## 📧 Contact

**Samadhi Jagathsiri** - [anusarasamm@gmail.com](mailto:your.email@example.com)

Project Link: [https://github.com/SamadhiJagathsiri/CalmeraAI_Chatbot](https://github.com/yourusername/CalmeraAI_Chatbot)

---

© 2025 Calmera AI. All Rights Reserved. | Developed by **Samadhi Jagathsiri**
