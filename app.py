import io
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from abalone_tools import FEATURES, NUMERIC, LIMITS, DEFAULTS, prepare

MM = 200.0
UNITS = {"Millimetres and grams": MM, "Dataset units (value / 200)": 1.0}
LENGTHS = {"Length", "Diameter", "Height"}
SAMPLE = {"Sex": "M", "Length": 0.455, "Diameter": 0.365, "Height": 0.095, "Whole_weight": 0.514,
          "Shucked_weight": 0.2245, "Viscera_weight": 0.101, "Shell_weight": 0.15}

st.set_page_config(page_title="Abalone Age Predictor", page_icon="🐚", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("abalone_model.joblib")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta


model, meta = load_artifacts()


def predict(data):
    rings = model.predict(data[FEATURES])
    bin_ = np.digitize(rings, meta["bins"][1:-1])
    out = pd.DataFrame({"Predicted_rings": rings.round(1), "Predicted_age_years": (rings + 1.5).round(1)}, index=data.index)
    out["Lower_80"] = np.maximum(rings + np.array(meta["lo"])[bin_], 1).round(1)
    out["Upper_80"] = (rings + np.array(meta["hi"])[bin_]).round(1)
    return out


def label(name, factor):
    if factor != MM:
        return name.replace("_", " ")
    return "%s (%s)" % (name.replace("_", " "), "mm" if name in LENGTHS else "g")


st.title("🐚 Abalone age predictor")
st.caption("Predicts shell rings from physical measurements. Estimated age in years = rings + 1.5.")

with st.sidebar:
    st.header("Model")
    st.write("**%s**, trained on %d cleaned abalone." % (meta["model"], meta["n_train"]))
    st.metric("Test MAE", "%.2f rings" % meta["test"]["MAE"])
    st.metric("Test RMSE", "%.2f rings" % meta["test"]["RMSE"])
    st.metric("Test R²", "%.2f" % meta["test"]["R2"])
    st.write("Within 1 ring: **%.0f%%**  \nWithin 2 rings: **%.0f%%**" % (100 * meta["within1"], 100 * meta["within2"]))
    st.caption("Single Tasmanian survey. Accuracy drops for old abalone (13+ rings), which are under-predicted.")

units = st.radio("Input units", list(UNITS), horizontal=True)
factor = UNITS[units]

tab1, tab2 = st.tabs(["Single abalone", "Excel / CSV upload"])

with tab1:
    sex = st.selectbox("Sex", ["M", "F", "I"], format_func=lambda s: {"M": "Male", "F": "Female", "I": "Infant"}[s])
    cols = st.columns(2)
    values = {}
    for i, name in enumerate(NUMERIC):
        lo, hi = LIMITS[name]
        values[name] = cols[i % 2].number_input(
            label(name, factor), min_value=lo * factor, max_value=hi * factor, value=DEFAULTS[name] * factor,
            step=0.1 if factor == MM else 0.005, format="%.2f" if factor == MM else "%.4f", key="%s_%s" % (name, factor))
    if st.button("Predict", type="primary"):
        data, errors = prepare(pd.DataFrame([{"Sex": sex, **values}]), factor, single=True)
        if errors:
            for e in errors:
                st.error(e)
        else:
            res = predict(data).iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted rings", "%.1f" % res["Predicted_rings"])
            c2.metric("Estimated age", "%.1f yr" % res["Predicted_age_years"])
            c3.metric("Likely range (80%)", "%.1f to %.1f" % (res["Lower_80"], res["Upper_80"]))
            st.caption("The range is the 10th to 90th percentile of cross-validated errors for abalone with a similar prediction.")

with tab2:
    st.write("Required columns: " + ", ".join(FEATURES))
    template = pd.DataFrame([{**SAMPLE, **{k: v * factor for k, v in SAMPLE.items() if k != "Sex"}}])
    buf = io.BytesIO()
    template.to_excel(buf, index=False)
    st.download_button("Download Excel template (%s)" % units.split(" (")[0].lower(), buf.getvalue(), "abalone_template.xlsx")
    up = st.file_uploader("Excel or CSV file", type=["xlsx", "xls", "csv"])
    if up is not None:
        try:
            raw_in = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        except Exception as e:
            st.error("Could not read the file: %s" % e)
        else:
            if len(raw_in) > 20000:
                st.error("Please upload at most 20,000 rows.")
            else:
                data, errors = prepare(raw_in, factor)
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    result = pd.concat([raw_in[FEATURES], predict(data)], axis=1)
                    st.success("%d abalone predicted." % len(result))
                    st.dataframe(result, use_container_width=True)
                    st.download_button("Download predictions (CSV)", result.to_csv(index=False).encode(),
                                       "abalone_predictions.csv", "text/csv")
