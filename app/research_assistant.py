
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def load_project_results():
    results_path = BASE_DIR / "results" / "model_comparison.csv"

    if not results_path.exists():
        return "Model comparison results are not available."

    df = pd.read_csv(results_path)

    # Keep the strongest configurations for a compact context.
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    available = [col for col in metrics if col in df.columns]

    if not available:
        return "The results file does not contain evaluation metrics."

    best = (
        df.sort_values("Accuracy", ascending=False)
        .groupby(["Method", "Components", "Model"], as_index=False)[available]
        .mean()
        .sort_values("Accuracy", ascending=False)
        .head(15)
    )

    return best.round(4).to_string(index=False)


def answer_question(question):
    if not API_KEY:
        return "Groq API key not found. Please check your .env file."

    try:
        client = Groq(api_key=API_KEY)
        project_results = load_project_results()

        system_prompt = f"""
You are an AI research assistant for a B.Tech major project titled:

Comparative Analysis of PCA and FastICA for
Dimensionality Reduction in Breast Cancer Classification.

PROJECT CONTEXT:
- Dataset: Wisconsin Diagnostic Breast Cancer dataset.
- Total samples: 569.
- Original features: 30 numerical tumour measurements.
- Target: Benign (B) and Malignant (M).
- Dimensionality reduction methods: PCA and FastICA.
- Classifiers: SVM, Logistic Regression, KNN and Random Forest.
- Evaluation: Accuracy, Precision, Recall, F1-score and ROC-AUC.
- Explainability: SHAP analysis on PCA-transformed components.

ACTUAL PROJECT RESULTS:
{project_results}

INSTRUCTIONS:
1. Answer using the project context and actual results wherever relevant.
2. Never invent experimental values or claim an experiment was performed
   if it is not shown in the supplied results.
3. Explain technical concepts in clear, student-friendly language.
4. Distinguish general scientific knowledge from project findings.
5. Help with research interpretation and university viva preparation.
6. Do not present model predictions as medical diagnoses.
7. If the available results do not answer a question, say so clearly.
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as error:
        return f"AI Assistant error: {error}"
