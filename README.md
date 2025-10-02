# Self-Healing Classification DAG

## 📌 Project Overview
This project demonstrates a **self-healing text classification pipeline** using **LangGraph**.  
We fine-tuned **DistilBERT** on the **IMDB Sentiment Analysis dataset** and integrated it into a **Directed Acyclic Graph (DAG)** with a **fallback strategy**.  

The system prioritizes correctness by:  
1. Running predictions with the fine-tuned model.  
2. Checking confidence scores.  
3. Falling back to a **backup zero-shot model** or **asking the user for clarification** when confidence is low.  

---

## 🎯 Features
- Fine-tuned transformer model (**DistilBERT on IMDB**)  
- **Three DAG nodes**:
  1. **InferenceNode** → Runs classification using the fine-tuned model  
  2. **ConfidenceCheckNode** → Checks confidence threshold (set to 0.75)  
  3. **FallbackNode** → Uses a **backup model** (`facebook/bart-large-mnli`) or **asks the user**  
- **CLI interface** for interactive classification  
- **Structured logging** to `pipeline.log`  

---

## ⚙️ Installation
Clone the repository and install dependencies:
```bash
git clone <your-repo-link>
cd <your-repo>
pip install -r requirements.txt
```

**requirements.txt:**
```
transformers
datasets
evaluate
accelerate
langgraph
torch
```

---

## 🚀 Fine-tuning the Model
The model is fine-tuned inside **`model_finetune.ipynb`**.  

Steps:
1. Open the notebook in Google Colab or Jupyter.  
2. Run all cells to:
   - Load IMDB dataset  
   - Fine-tune DistilBERT  
   - Save the model in `fine_tuned_imdb/`  
3. The trained model is then used by the pipeline.  

---

## 🚀 Running the LangGraph Pipeline
Run the CLI interface with:
```bash
python pipeline.py
```

You will see:
```
Sentiment Classification CLI with Backup Fallback (type 'quit' to exit)
```

Enter any movie review and the pipeline will process it.

---

## 🖥 Example CLI Flow

### ✅ Case 1: Normal Prediction
```
Enter a review: I absolutely loved this movie, it was fantastic!
Final Label: POSITIVE (main conf: 0.92, backup: N/A (0.00))
```

### 🔄 Case 2: Fallback → Backup Model Correction
```
Enter a review: The visuals were stunning, but the plot made no sense.
[FallbackNode] Backup Model suggests: NEGATIVE (conf: 0.81)
[FallbackNode] Main model = POSITIVE (conf: 0.62), Backup model = NEGATIVE (0.81)
Final Label: NEGATIVE (main conf: 0.62, backup: NEGATIVE (0.81))
```

### 👤 Case 3: Fallback → User Clarification
```
Enter a review: The acting was brilliant, but the story was dull and predictable.
[FallbackNode] Backup Model suggests: NEGATIVE (conf: 0.69)
[FallbackNode] Main model = POSITIVE (conf: 0.58), Backup model = NEGATIVE (0.69)
They disagree. Please clarify (POSITIVE/NEGATIVE): NEGATIVE
Final Label: NEGATIVE (main conf: 0.58, backup: NEGATIVE (0.69))
```

---

## 📝 Logs
All predictions and fallbacks are logged in **`pipeline.log`**.  

Example log snippet:
```
2025-10-02 18:12:33 - INFO - [InferenceNode] Input: The movie was okay, not too good, not too bad. | Predicted: POSITIVE | Confidence: 0.56
2025-10-02 18:12:33 - INFO - [ConfidenceCheckNode] Low confidence (0.56). Triggering fallback...
2025-10-02 18:12:35 - INFO - [FallbackNode] Backup Model Prediction: POSITIVE | Confidence: 0.65
2025-10-02 18:12:35 - INFO - [FallbackNode] Backup agrees with main model → Accepting POSITIVE
```

---

## 🎥 Demo Video
The demo video (2–4 minutes) shows:  
1. Normal prediction  
2. Backup model correction  
3. User clarification  

---

## 🔮 Future Improvements
- Add multiple backup models (ensembling)  
- Show fallback statistics (e.g., histogram of fallback frequency)  
- Deploy as a **web app** for non-technical users  

---
