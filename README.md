---
title: Multi-Modal RAG
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🎯 Multi-Modal RAG: Documents That See

**Chat with documents containing text, charts, and images.**

Upload any PDF → Ask questions about text AND visuals → Get cited answers.

## 🌐 Live Demo

This Space is the live demo! Upload a PDF and start chatting.

## 🎥 What It Does

Upload a research paper, financial report, or any document with figures:

```
❓ "What is the main architecture shown in Figure 3?"
❓ "Explain the trend in the bar chart on page 7"
❓ "Summarize the key findings from the results section"
```

## 🚀 Run Locally

```bash
git clone https://github.com/vraul92/multimodal-rag.git
cd multimodal-rag
pip install -r requirements.txt
python app.py
```

## 🛠️ Tech Stack

- **Frontend**: Gradio 4.0+
- **Text Embeddings**: BAAI/bge-m3
- **Visual Embeddings**: OpenAI CLIP
- **Vector Store**: FAISS
- **PDF Processing**: PyMuPDF

## 🤝 Author

**Rahul Vuppalapati** - Senior Data Scientist
- Previously: Apple, Walmart, IBM
- GitHub: https://github.com/vraul92
- LinkedIn: https://linkedin.com/in/vrc7

## 📄 License

MIT License
