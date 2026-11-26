Explainable Machine Learning Platform for Breast Cancer Research

OncoLens is an interactive machine learning research platform designed to explore breast cancer datasets, experiment with classification models, evaluate predictive performance, and interpret model decisions using Explainable AI (XAI).

Built with Python and Streamlit, the platform combines machine learning experimentation, dimensionality reduction, SHAP-based interpretability, and an AI-powered research assistant in one interface.

✨ Key Features

1. Dataset Explorer

Upload CSV and Excel datasets.

Explore dataset dimensions, data types, and statistical summaries.

Identify missing values and empty columns.

Visualize feature distributions and dataset characteristics.

2. Model Laboratory

Select target and input features.

Apply dimensionality reduction techniques such as PCA and FastICA.

Experiment with multiple classification algorithms.

Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.

Compare model performance to identify the most suitable configuration.

3. Explainable AI

Interpret model predictions using SHAP.

Examine global feature importance.

Visualize how individual features contribute to model predictions.

Explore the factors influencing classification outcomes.

4. AI Research Assistant

Interact with an AI-powered assistant for research-related queries.

Generate explanations based on dataset and model evaluation context.

Powered by the Groq API.

🛠️ Tech Stack

Category

Technologies

Programming

Python

Interface

Streamlit

Data Processing

Pandas, NumPy

Machine Learning

Scikit-learn

Explainable AI

SHAP

Visualization

Plotly, Matplotlib

AI Integration

Groq API

Environment Management

uv

📁 Project Structure

OncoLens/
│
├── app/
│   ├── app.py
│   ├── ml_engine.py
│   └── research_assistant.py
│
├── data/
│   └── data.csv
│
├── src/
│   ├── __init__.py
│   └── breastcancer_research/
│
├── .gitignore
├── pyproject.toml
├── uv.lock
└── requirements.txt

⚙️ Installation and Setup

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/OncoLens.git
cd OncoLens

2. Install dependencies

Using uv:

uv sync

3. Configure environment variables

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key

Obtain your API key from Groq Console.

4. Run the application

uv run streamlit run app/app.py

The application will open in your browser.

📊 Research Focus

OncoLens uses breast cancer data to investigate machine learning-based classification and model interpretability.

The platform focuses on:

Comparing classification approaches.

Understanding the effect of dimensionality reduction.

Evaluating predictive performance.

Interpreting model behaviour through SHAP.

⚠️ Disclaimer

OncoLens is an academic research and experimentation project. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical diagnosis or treatment.

👩‍💻 Author

Shefali Kushwah

B.Tech Biotechnology | NIT Jalandhar

GitHub

Exploring the intersection of biotechnology, machine learning, and explainable artificial intelligence.
