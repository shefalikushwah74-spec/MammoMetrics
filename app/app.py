import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import shap
import matplotlib.pyplot as plt

from ml_engine import run_experiments

APP_NAME = "MammoSense"   # <- naam yahin se change hota hai
TAGLINE = "Benign vs malignant tumour classification with explainable AI"

st.set_page_config(page_title=f"{APP_NAME} | Breast tumour classification",
                   page_icon="🎗️", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------
# DESIGN TOKENS
# ---------------------------------------------------------
ROSE, ROSE_DARK, ROSE_SOFT = "#B83265", "#962753", "#FCEBF1"     # malignant / primary
TEAL, TEAL_DARK, TEAL_SOFT = "#0E7C74", "#0A5F59", "#E2F3F1"     # benign / secondary
INK, MUTED, LINE, CANVAS = "#1F2937", "#566170", "#EBE1E6", "#FCFAFB"
AMBER = "#C27A1A"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [class*="css"] {{ font-family:'Figtree',sans-serif; font-size:16px; color:{INK}; }}
.stApp {{ background:{CANVAS}; }}
[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ max-width:1280px; padding-top:2rem; padding-bottom:3rem; }}

/* type scale: 30 / 21 / 17 / 16 body / 14 caption */
h1 {{ font-family:'Figtree'!important; font-size:1.9rem!important; font-weight:800!important; color:{INK}!important; letter-spacing:-0.02em; margin-bottom:.2rem; }}
h2, h3 {{ font-family:'Figtree'!important; color:{INK}!important; letter-spacing:-0.01em; }}
h2 {{ font-size:1.3rem!important; font-weight:700!important; }}
h3 {{ font-size:1.08rem!important; font-weight:700!important; }}
p, li {{ font-size:1rem; line-height:1.6; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color:{MUTED}!important; font-size:.875rem!important; }}
label, [data-testid="stWidgetLabel"] p {{ color:{INK}!important; font-weight:600!important; font-size:.95rem!important; }}

/* sidebar */
[data-testid="stSidebar"] {{ background:#FFFFFF; border-right:1px solid {LINE}; }}
[data-testid="stSidebar"] .stRadio > div {{ gap:.25rem; }}
[data-testid="stSidebar"] .stRadio label {{ padding:.55rem .8rem; border-radius:10px; width:100%; }}
[data-testid="stSidebar"] .stRadio label:hover {{ background:{ROSE_SOFT}; }}
[data-testid="stSidebar"] .stRadio label p {{ font-size:.98rem; font-weight:600; color:{INK}; }}

.brand {{ display:flex; align-items:center; gap:.6rem; margin-bottom:.2rem; }}
.brand-mark {{ width:34px; height:34px; border-radius:10px; background:{ROSE}; color:#fff; display:flex; align-items:center; justify-content:center; font-size:1.1rem; }}
.brand-name {{ font-size:1.35rem; font-weight:800; color:{INK}; letter-spacing:-0.02em; }}
.brand-sub {{ color:{MUTED}; font-size:.85rem; line-height:1.4; margin-bottom:.4rem; }}
.page-sub {{ color:{MUTED}; font-size:1.02rem; max-width:760px; margin:0 0 1.2rem 0; }}

/* metric cards */
[data-testid="stMetric"] {{ background:#fff; border:1px solid {LINE}; border-radius:14px; padding:16px 18px; }}
[data-testid="stMetricLabel"] p {{ color:{MUTED}!important; font-size:.88rem!important; font-weight:600!important; }}
[data-testid="stMetricValue"] {{ color:{INK}!important; font-weight:800!important; font-size:1.7rem!important; }}

/* buttons */
.stButton > button {{ background:{ROSE}; color:#fff; border:1px solid {ROSE}; border-radius:10px; padding:.6rem 1.2rem; font-weight:700; font-size:.98rem; }}
.stButton > button:hover {{ background:{ROSE_DARK}; border-color:{ROSE_DARK}; color:#fff; }}
.stButton > button:focus {{ box-shadow:0 0 0 .2rem rgba(184,50,101,.22); }}

div[data-testid="stFileUploader"] section {{ background:#fff; border:1.5px dashed #D9BFCA; border-radius:14px; }}
div[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:12px; overflow:hidden; }}
[data-testid="stExpander"] {{ background:#fff; border:1px solid {LINE}; border-radius:12px; }}
hr {{ border-color:{LINE}; }}
footer {{ visibility:hidden; }}

/* callouts */
.callout {{ border-radius:12px; padding:.9rem 1.1rem; margin:.4rem 0 1rem 0; border:1px solid; font-size:.98rem; line-height:1.55; }}
.callout b {{ font-weight:700; }}
.callout.rose {{ background:{ROSE_SOFT}; border-color:#F3C7D6; color:#5E1431; }}
.callout.teal {{ background:{TEAL_SOFT}; border-color:#B7DEDA; color:#0A4642; }}
.callout.amber {{ background:#FDF3E3; border-color:#F0D9B0; color:#5C3A06; }}
.pill {{ display:inline-block; padding:.22rem .7rem; border-radius:999px; background:{TEAL_SOFT}; color:{TEAL_DARK}; font-size:.85rem; font-weight:700; }}
.pill.rose {{ background:{ROSE_SOFT}; color:{ROSE_DARK}; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def class_color(name):
    n = str(name).strip().lower()
    if n in ("m", "malignant", "1", "yes", "positive"):
        return ROSE
    if n in ("b", "benign", "0", "no", "negative"):
        return TEAL
    return "#6B7A99"


def color_map(values):
    return {str(v): class_color(v) for v in values}


def style_plot(fig, height=380):
    fig.update_layout(
        height=height, template="plotly_white",
        font=dict(family="Figtree", color=INK, size=14),
        title_font=dict(family="Figtree", size=17, color=INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        margin=dict(l=16, r=16, t=56, b=16), legend_title_text="",
        legend=dict(font=dict(size=13)),
    )
    fig.update_xaxes(showgrid=False, linecolor="#D8CDD3", title_font=dict(size=13), tickfont=dict(size=12))
    fig.update_yaxes(gridcolor="#F1EAEE", zeroline=False, title_font=dict(size=13), tickfont=dict(size=12))
    return fig


def page_heading(title, description):
    st.title(title)
    st.markdown(f'<p class="page-sub">{description}</p>', unsafe_allow_html=True)


def callout(text, tone="teal"):
    st.markdown(f'<div class="callout {tone}">{text}</div>', unsafe_allow_html=True)


def likely_identifier_columns(columns):
    keys = ("id", "index", "uuid", "patient_id", "sample_id", "record_id", "serial")
    out = []
    for c in columns:
        n = str(c).strip().lower()
        if n in keys or any(n.endswith("_" + k) for k in keys):
            out.append(c)
    return out


def label_candidates(df):
    cols = []
    for c in df.columns:
        if df[c].nunique(dropna=True) <= 6 and not str(c).lower().endswith("id"):
            cols.append(c)
    return cols


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown(
    f'<div class="brand"><div class="brand-mark">🎗️</div><div class="brand-name">{APP_NAME}</div></div>'
    f'<div class="brand-sub">{TAGLINE}</div>', unsafe_allow_html=True)
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Tumour data", "Model comparison", "Why the model decides", "Ask the assistant"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption("Research prototype. Not a diagnostic tool.")

for key in ("df", "experiment", "dataset_name", "shap_values", "shap_class"):
    st.session_state.setdefault(key, None)


# ---------------------------------------------------------
# 1. TUMOUR DATA
# ---------------------------------------------------------
if page == "Tumour data":
    page_heading("Tumour data",
                 "Upload your tumour measurements, check the diagnosis split and see which features separate benign from malignant cases.")

    uploaded = st.file_uploader("Upload dataset (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            empty = df.columns[df.isna().all()].tolist()
            if empty:
                df = df.drop(columns=empty)
                st.info(f"Removed empty columns: {', '.join(map(str, empty))}")
            st.session_state.update(df=df, dataset_name=uploaded.name, experiment=None, shap_values=None)
        except Exception as e:
            st.error(f"Could not read the file: {e}")

    df = st.session_state.df
    if df is None:
        callout("<b>No dataset yet.</b> Upload the Wisconsin breast cancer CSV (or a similar file with a diagnosis column) to begin.", "amber")
    else:
        st.markdown(f'<span class="pill">Loaded</span> &nbsp; {st.session_state.dataset_name}', unsafe_allow_html=True)
        st.write("")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        total_missing = int(df.isna().sum().sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients / samples", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Measurement features", f"{len(numeric_cols):,}")
        c4.metric("Missing values", f"{total_missing:,}")

        cands = label_candidates(df)
        default_label = next((c for c in cands if str(c).lower() in ("diagnosis", "target", "class", "label")), cands[0] if cands else None)
        label = None
        if cands:
            label = st.selectbox("Diagnosis column", cands, index=cands.index(default_label),
                                 help="Used only to colour the charts below. Choose it again for training on the next page.")

        st.write("")
        left, right = st.columns([1.1, 0.9], gap="large")

        with left:
            st.subheader("Sample records")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        with right:
            st.subheader("Benign vs malignant")
            if label is not None:
                counts = df[label].astype(str).value_counts().reset_index()
                counts.columns = ["Diagnosis", "Cases"]
                fig = px.pie(counts, names="Diagnosis", values="Cases", hole=0.62,
                             color="Diagnosis", color_discrete_map=color_map(counts["Diagnosis"]))
                fig.update_traces(textinfo="percent+label", textfont_size=14,
                                  marker=dict(line=dict(color="#fff", width=3)))
                fig.update_layout(showlegend=False,
                                  annotations=[dict(text=f"<b>{len(df):,}</b><br>cases", showarrow=False, font=dict(size=17))])
                st.plotly_chart(style_plot(fig, 310), use_container_width=True)
                minority = counts["Cases"].min() / counts["Cases"].sum()
                if minority < 0.3:
                    callout("The classes are imbalanced. Check recall for the malignant class, not just accuracy.", "amber")
            else:
                st.caption("No diagnosis-like column found.")

        st.divider()
        st.subheader("How a feature differs between diagnoses")
        if numeric_cols:
            feat = st.selectbox("Measurement", numeric_cols)
            fig = px.histogram(df, x=feat, color=label if label else None, nbins=30, opacity=0.72,
                               barmode="overlay", marginal="box",
                               color_discrete_map=color_map(df[label].astype(str).unique()) if label else None)
            if label:
                fig.for_each_trace(lambda t: t.update(name=str(t.name)))
            fig.update_yaxes(title_text="Cases", row=1, col=1)
            st.plotly_chart(style_plot(fig, 400), use_container_width=True)
            st.caption("Little overlap between the two colours means the feature separates the classes well.")

        if label is not None and len(numeric_cols) >= 3:
            st.divider()
            st.subheader("Features most linked to diagnosis")
            id_cols = likely_identifier_columns(df.columns)
            usable = [c for c in numeric_cols if c not in id_cols and c != label]
            y = pd.factorize(df[label])[0]
            corr_y = df[usable].apply(lambda s: np.corrcoef(s.fillna(s.median()), y)[0, 1]).abs().sort_values(ascending=False)
            top = corr_y.head(10).index.tolist()
            cm_ = df[top].corr()
            fig_h = px.imshow(cm_, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                              color_continuous_scale=[[0, TEAL], [0.5, "#FFFFFF"], [1, ROSE]])
            fig_h.update_layout(coloraxis_colorbar=dict(title="Correlation", thickness=12))
            st.plotly_chart(style_plot(fig_h, 480), use_container_width=True)
            st.caption("Top 10 features by link to diagnosis. Highly correlated features carry overlapping information.")

        with st.expander("Column details"):
            info = pd.DataFrame({"Column": df.columns.astype(str), "Type": df.dtypes.astype(str).values,
                                 "Missing": df.isna().sum().values, "Unique values": df.nunique().values})
            st.dataframe(info, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# 2. MODEL COMPARISON
# ---------------------------------------------------------
elif page == "Model comparison":
    page_heading("Model comparison",
                 "Train several classifiers on the original and reduced features, then compare how well each one finds malignant tumours.")

    df = st.session_state.df
    if df is None:
        callout("<b>Upload a dataset first.</b> Go to <b>Tumour data</b> in the sidebar.", "amber")
    else:
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        target = st.selectbox("Diagnosis column", cat_cols or df.columns.tolist(),
                              help="For the Wisconsin dataset choose 'diagnosis' (B = benign, M = malignant).")

        id_c = [c for c in likely_identifier_columns(df.columns) if c != target]
        num_c = [c for c in df.select_dtypes(include=np.number).columns if c != target]
        excluded = st.multiselect("Leave out ID or irrelevant columns", [c for c in df.columns if c != target],
                                  default=[c for c in id_c if c in num_c],
                                  help="Patient IDs should not be used to predict a diagnosis.")
        feats_avail = [c for c in num_c if c not in excluded]
        features = st.multiselect("Measurements to use", feats_avail, default=feats_avail)
        components = st.multiselect("Reduced feature counts (PCA / FastICA)", [2, 5, 10, 15, 20], default=[2, 5, 10])

        with st.expander("What gets trained"):
            st.markdown("- **Inputs:** original features, PCA, FastICA\n"
                        "- **Classifiers:** Logistic Regression, SVM (RBF), Random Forest\n"
                        "- **Checks:** stratified cross-validation, then one held-out test set\n"
                        "- **Metrics:** accuracy, precision, recall, F1")

        run_col, note_col = st.columns([0.3, 0.7], vertical_alignment="center")
        run = run_col.button("Train and compare models", use_container_width=True)
        note_col.caption("This can take a minute on larger files.")

        if run:
            if len(features) < 2:
                st.error("Pick at least two measurements.")
            else:
                try:
                    with st.spinner("Training models..."):
                        exp = run_experiments(df, target=target, feature_columns=features,
                                              components=tuple(components) if components else (2, 5, 10))
                    st.session_state.experiment = exp
                    st.session_state.shap_values = None
                    st.success("Training finished.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

        exp = st.session_state.experiment
        if exp is not None:
            st.divider()
            best = exp["best_configuration"]
            st.subheader("Best setup")
            b1, b2, b3 = st.columns(3)
            b1.metric("Input features", str(best.get("Method", "—")))
            b2.metric("Classifier", str(best.get("Model", "—")))
            b3.metric("Reduced features", str(best.get("Components", "—")))

            st.subheader("Results on unseen test cases")
            m = exp["test_metrics"]
            cols = st.columns(4)
            hints = {"Accuracy": "Overall correct", "Precision": "Of predicted malignant, truly malignant",
                     "Recall": "Of true malignant, found", "F1": "Balance of precision and recall"}
            for col, k in zip(cols, ["Accuracy", "Precision", "Recall", "F1"]):
                v = m.get(k)
                col.metric(k, f"{v:.1%}" if v is not None else "—", help=hints[k])

            # confusion matrix + missed-cancer callout
            st.divider()
            cm = np.array(exp["confusion_matrix"])
            names = [str(n) for n in exp["class_names"]]
            mal = next((i for i, n in enumerate(names) if class_color(n) == ROSE), len(names) - 1)
            fn = int(cm[mal].sum() - cm[mal, mal])
            fp = int(cm[:, mal].sum() - cm[mal, mal])

            g1, g2 = st.columns([1.1, 0.9], gap="large")
            with g1:
                st.subheader("Prediction outcomes")
                fig_cm = go.Figure(go.Heatmap(
                    z=cm, x=names, y=names, text=cm, texttemplate="%{text}", textfont=dict(size=22),
                    colorscale=[[0, "#FFFFFF"], [1, TEAL]], showscale=False, xgap=4, ygap=4))
                fig_cm.update_layout(xaxis_title="Model predicted", yaxis_title="Actual diagnosis",
                                     yaxis=dict(autorange="reversed"))
                st.plotly_chart(style_plot(fig_cm, 360), use_container_width=True)
            with g2:
                st.subheader("What this means")
                callout(f"<b>{fn} malignant case(s) missed</b> (predicted as non-malignant). This is the most serious error type.",
                        "rose" if fn else "teal")
                callout(f"<b>{fp} false alarm(s)</b>: predicted malignant, actually not. These lead to extra tests, not missed disease.", "amber")
                st.caption("Aim to keep missed malignant cases as low as possible, even at the cost of a few false alarms.")

            st.divider()
            st.subheader("All models, side by side")
            res = exp["results"].copy().sort_values("F1", ascending=False)
            fig_c = px.bar(res, x="Method", y="F1", color="Model", barmode="group",
                           hover_data=[c for c in ["Components", "Accuracy", "Precision", "Recall"] if c in res.columns],
                           color_discrete_sequence=[ROSE, TEAL, AMBER], title="Cross-validated F1 by input type and classifier")
            fig_c.update_yaxes(tickformat=".0%", rangemode="tozero", title_text="F1 score")
            st.plotly_chart(style_plot(fig_c, 420), use_container_width=True)

            with st.expander("Full results table"):
                st.dataframe(res.style.format({c: "{:.1%}" for c in ["Accuracy", "Precision", "Recall", "F1"] if c in res.columns}),
                             use_container_width=True, hide_index=True)
            with st.expander("Detailed classification report"):
                st.code(exp["classification_report"], language="text")

            callout("High accuracy alone does not make a model clinically useful. This is a research and learning prototype, not a diagnostic tool.", "amber")


# ---------------------------------------------------------
# 3. WHY THE MODEL DECIDES (SHAP)
# ---------------------------------------------------------
elif page == "Why the model decides":
    page_heading("Why the model decides",
                 "See which tumour measurements pushed the model towards a malignant or benign prediction.")

    exp = st.session_state.experiment
    if exp is None:
        callout("<b>Train a model first.</b> Open <b>Model comparison</b> and run the training.", "amber")
    else:
        model, X_train, X_test = exp["model"], exp["X_train"], exp["X_test"]
        class_names = list(exp["class_names"])

        callout("SHAP shows how much each measurement moved a prediction. It explains the model's behaviour, not the biology of the disease.", "teal")

        sel = st.selectbox("Explain predictions for class", class_names,
                           index=class_names.index("M") if "M" in class_names else 0)
        ci = class_names.index(sel)
        n = st.slider("Test cases to explain", 5, max(5, min(30, len(X_test))), min(15, max(5, len(X_test))))

        if st.button("Generate explanations"):
            try:
                with st.spinner("Calculating SHAP values..."):
                    bg = shap.sample(X_train, min(30, len(X_train)), random_state=42)
                    data = X_test.head(min(n, len(X_test)))
                    explainer = shap.Explainer(model.predict_proba, bg, algorithm="permutation")
                    sv = explainer(data, max_evals=2 * X_train.shape[1] + 1)
                    st.session_state.shap_values = sv[:, :, ci] if len(sv.shape) == 3 else sv
                    st.session_state.shap_class = sel
                st.success("Explanations ready.")
            except Exception as e:
                st.session_state.shap_values = None
                st.error(f"Could not generate explanations: {e}")

        sv = st.session_state.shap_values
        if sv is not None:
            if st.session_state.shap_class != sel:
                st.warning("You changed the class. Generate explanations again.")
            else:
                st.divider()
                st.subheader("Most influential measurements")
                st.caption("Average impact on the prediction, across the selected cases.")
                plt.figure(figsize=(9, 5))
                shap.plots.bar(sv, max_display=15, show=False)
                st.pyplot(plt.gcf(), clear_figure=True)
                plt.close("all")

                st.subheader("Impact on each case")
                st.caption("Each dot is one case. Colour shows whether the measurement value was high or low.")
                plt.figure(figsize=(9, 6))
                shap.plots.beeswarm(sv, max_display=15, show=False)
                st.pyplot(plt.gcf(), clear_figure=True)
                plt.close("all")

                callout(f"Values to the right push the prediction <b>towards '{sel}'</b>; values to the left push it away.", "teal")


# ---------------------------------------------------------
# 4. ASSISTANT
# ---------------------------------------------------------
elif page == "Ask the assistant":
    page_heading("Ask the assistant",
                 "Ask about your results, metrics like recall, PCA vs FastICA, or how to read the SHAP charts.")

    exp = st.session_state.experiment
    if exp is None:
        callout("You can ask general questions now. Train a model to let the assistant see your results.", "amber")
    else:
        st.markdown('<span class="pill">Your results are connected</span>', unsafe_allow_html=True)
        st.write("")

    q = st.text_area("Your question", height=120,
                     placeholder="e.g. Why did PCA with 10 components perform better than FastICA?")

    if st.button("Ask"):
        if not q.strip():
            st.warning("Type a question first.")
        else:
            try:
                from research_assistant import answer_question
                ctx = "Project: Breast tumour classification research prototype."
                if exp is not None:
                    ctx += (f"\nBest configuration: {exp['best_configuration']}"
                            f"\nHeld-out test metrics: {exp['test_metrics']}"
                            f"\nModel comparison:\n{exp['results'].to_string(index=False)}"
                            f"\nClass labels: {list(exp['class_names'])}")
                with st.spinner("Thinking..."):
                    ans = answer_question(q + "\n\nUse the following project context when relevant. "
                                          "Do not invent results or claim clinical validity.\n" + ctx)
                st.divider()
                st.subheader("Answer")
                st.markdown(ans)
            except Exception as e:
                st.error(f"Assistant error: {e}")

st.divider()
st.caption(f"{APP_NAME} · Academic research prototype · Not for diagnosis or treatment decisions")