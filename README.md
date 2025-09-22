# 📘 Automated MCQ Generation (Savaal-inspired)

This project implements an **automated multiple-choice question generation system** that converts raw documents (PDF or TXT) into structured, difficulty-grouped questions.  
It is inspired by the design principles of the [Savaal research paper](https://arxiv.org/abs/2305.17859).  

---

## ✨ Features
- **Input Support**: Accepts both PDF and TXT files.
- **Adaptive Chunking**: Splits large text intelligently to manage memory & API costs.
- **Three-Step Pipeline**:
  1. **Main Idea Extraction** – High-level concept extraction in batches.  
  2. **Relevant Passage Retrieval** – Uses FAISS vector search with HuggingFace embeddings.  
  3. **MCQ Generation** – Generates conceptual questions with distractors using Gemini API.  
- **Difficulty Grouping**: Questions are categorized into **Easy**, **Medium**, and **Hard** following Bloom’s taxonomy.  
- **Scalability**:  
  - Batching to minimize API calls.  
  - API budget control (`budget` parameter).  
  - Error handling for robustness.  
- **Quality Control**:  
  - Enforces strict JSON schema.  
  - Filters duplicates, trivial, or invalid questions.  

---

## 🏗️ Architecture

Input (PDF/TXT)
↓
Adaptive Chunking
↓
Main Idea Extraction (Gemini)
↓
Relevant Passage Retrieval (FAISS + HuggingFace Embeddings)
↓
Question Generation (Gemini → JSON Schema)
↓
Filtering & Grouping
↓
Final Output (questions.json)

```bash

## 📂 Output Format
All results are saved in `questions.json` with grouped difficulty levels:

```json
{
  "questions": {
    "easy": [
      {
        "question": "What is the definition of X?",
        "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
        "answer": "B. ..."
      }
    ],
    "medium": [...],
    "hard": [...]
  }
}
```


# ⚡ Installation & Setup
# 1. Clone the repo
git clone https://github.com/yourusername/mcq-generator.git
cd mcq-generator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API key

This project uses Google Gemini API.
Store your key as an environment variable:

export GOOGLE_API_KEY="your_api_key_here"


# 4. Run the script


🚀 Example Run
FILE_PATH = "/Documents/transcript_1.txt"
results = run_pipeline(FILE_PATH, questions_per_idea=2, max_api_calls=20, output_file="questions.json")


Output:

✅ Pipeline complete. 45 high-quality questions saved to questions.json
