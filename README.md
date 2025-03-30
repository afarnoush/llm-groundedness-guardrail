# RAG-Based Chatbot Evaluation Framework

This project provides a robust evaluation framework to benchmark the performance of Retrieval-Augmented Generation (RAG) chatbots in a banking context. It enables comparisons across different LLM models, prompt strategies, and retrievers using real policy questions and document-grounded answers.

## 📌 Objectives
- Ensure chatbot responses are accurate, grounded, and aligned with policy.
- Compare multiple prompts and models using standard NLP metrics.
- Establish guardrails for safe and compliant chatbot deployment.

## 🧱 Project Structure
```
├── docsModel/                              # Markdown source documents for retriever
├── Prompts/                                # Prompt templates (.txt)
├── src/
│   ├── rag_evaluator.py                     # Main evaluation function
│   ├── retriever_builder.py                 # Ensemble retriever setup from markdown
│   └── utils.py                             # Shared utilities (BLEU, ROUGE, cosine, etc.)
├── Outputs/                                 # (Optional) Saved evaluation results
└── evaluation_notebook.ipynb                # Run experiments and visualize results
└── CBA_Chatbot_Evaluation_30_Questions.csv  # Evaluation CSV files (questions + expected answers)
```

## ⚙️ Features
- **Evaluate LLMs** (e.g. Gemma, Phi3) on chatbot response quality
- **Compare prompts**: default, strict grounding, sales-focused, structured
- **Support for multiple metrics**:
  - Cosine Similarity (semantic match)
  - BLEU (precision)
  - ROUGE (recall)
  
## 🚀 How to Use
1. Place your evaluation questions in `your_questions.csv`
2. Save product/policy documents in markdown format under `docsModel/`
3. Add or edit prompt templates in `Prompts/`
4. Run `evaluation_notebook.ipynb` to:
   - Load prompts + models
   - Run `evaluate_rag()`
   - Generate metrics
   - Visualize with bar charts

## 📊 Outputs
Each evaluation run outputs a DataFrame (and optional `.csv`) with:
- Generated answer
- Cosine similarity to expected
- BLEU / ROUGE / Recall@K
- Prompt and model used
- Time taken per query

## 🛡️ Guardrail Strategy
- Use Cosine Similarity to detect ungrounded responses
- Flag responses below a similarity threshold (e.g., < 0.6)
- Quantify and compare performance before deployment

## 📈 Example Visualizations
- Bar charts comparing BLEU, ROUGE, and Cosine across prompts and models

## ✅ Ideal For
- Risk and compliance testing of generative AI in banking
- Model and prompt A/B testing
- Safe scaling of RAG-based virtual assistants

## 👤 Author
Built by Azadeh Farnoush as part of a senior analytics innovation project.

