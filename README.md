# 🎙️ SonicSense — Multimodal Sentiment Analysis

> Fusing BERT text analysis with Wav2Vec 2.0 audio through cross-attention late fusion — with sarcasm detection, emotion trajectory over time, full SHAP explainability, and noise robustness benchmarking.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/License-MIT-purple)

---

##  What is this?

SonicSense is a **multimodal sentiment analysis system** that analyzes both **what you say** (text) and **how you say it** (audio) to predict emotion more accurately than either modality alone.

It was built as a deep learning project combining:
- **BERT** fine-tuned on the MELD dataset for text sentiment
- **Wav2Vec 2.0 CNN** fine-tuned on the same dataset for audio sentiment
- **Cross-Attention Late Fusion** to dynamically weight both modalities
- **SHAP** for token-level explainability
- **FastAPI** backend served from Google Colab via localtunnel
- **Vanilla HTML/JS** frontend with Chart.js visualizations

---

##  Features

| Feature | Description |
|---------|-------------|
|  Sarcasm Detection | Detects when text sentiment contradicts audio tone |
|  Emotion Trajectory | Sliding window inference over audio to show sentiment changing over time |
|  SHAP · XAI | Per-token attribution scores showing which words drove the prediction |
|  Cross-Modal Attention | Dynamic weighting of text vs audio per sample |
|  Noise Robustness | Benchmark chart showing model accuracy vs SNR across noise types |

---

##  Project Structure

```
SonicSense/
│
├── multimodal-sentiment-analyzer.html   # Frontend — open in browser
├── multimodal.ipynb                     # Google Colab backend notebook
├── requirements.txt                     # Python dependencies
└── README.md
```

---

##  How to Run

### Step 1 — Open the Colab notebook
Open `multimodal.ipynb` in Google Colab. Make sure you have a **T4 GPU** runtime enabled:
`Runtime → Change runtime type → T4 GPU`

### Step 2 — Run cells in order

| Cell | What it does |
|------|-------------|
| 1 | Install all dependencies |
| 2 | Set device + config |
| 3 | Load & preprocess MELD dataset |
| 4 | Fine-tune BERT (text model) |
| 5 | Fine-tune Wav2Vec2 (audio model) |
| 6 | Train cross-attention fusion layer |
| 7 | Evaluate all three models |
| 8 | Save models to Google Drive |
| 9 | Launch FastAPI server + localtunnel |

### Step 3 — Update the URL in the HTML
After running the final cell, you'll see:
```
your url is: https://xxxx-xxxx-xxxx.loca.lt
YOUR TUNNEL PASSWORD IS: XX.XX.XX.XX
```

Open `multimodal-sentiment-analyzer.html`, find the `analyze()` function and update:
```javascript
const res = await fetch('https://YOUR-URL.loca.lt/analyze', {
```

### Step 4 — Open the frontend
Open `multimodal-sentiment-analyzer.html` in your browser and click **Analyze Sentiment**.

---

##  Subsequent Sessions (Skip Training)

Once models are saved to Google Drive, you only need to run:
1. Install cell
2. Config cell
3. Model reload cell (loads from Drive)
4. Final server cell

**Total startup time: ~4 minutes**

---

##  Model Architecture

```
Text Input ──→ BERT (fine-tuned) ──→ Text Logits ──┐
                                                     ├──→ Cross-Attention Fusion ──→ Final Sentiment
Audio Input ──→ Wav2Vec2 CNN ──→ Audio Logits ───────┘
```

- **Dataset**: MELD (Multimodal EmotionLines Dataset) — 13k utterances from Friends TV show
- **Labels**: 4-class sentiment (positive, negative, neutral, mixed)
- **Fusion**: Cross-attention with dynamic per-sample modality weighting
- **Explainability**: SHAP partition explainer on BERT tokenizer

---

##  Results

| Model | Accuracy |
|-------|----------|
| Text-only (BERT) | ~76% |
| Audio-only (Wav2Vec2) | ~65% |
| Fusion (combined) | ~80% |

*Results vary based on training sample size and epochs.*

---

##  Tech Stack

**Backend**
- PyTorch 2.2.1
- HuggingFace Transformers 4.40.0
- Wav2Vec2 (facebook/wav2vec2-base)
- BERT (bert-base-uncased)
- FastAPI + Uvicorn
- SHAP 0.45.0
- librosa, soundfile, audiomentations

**Frontend**
- Vanilla HTML/CSS/JavaScript
- Chart.js 4.4.1
- Google Fonts (Syne, DM Mono, DM Sans)

**Infrastructure**
- Google Colab (T4 GPU)
- localtunnel (public URL)
- Google Drive (model persistence)

---

##  Dataset

This project uses the **MELD dataset** (Multimodal EmotionLines Dataset):
- Paper: [MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation](https://arxiv.org/abs/1810.02508)
- HuggingFace: `declare-lab/meld` / `Vano04/MELD-Preprocessed`

---

##  License

MIT License — feel free to use and modify.

---

##  Author

Built by **aryan12375**

