import io
import joblib
import pandas as pd
import streamlit as st

FEATURES = ["Sex", "Length", "Diameter", "Height", "Whole_weight",
            "Shucked_weight", "Viscera_weight", "Shell_weight"]
LIMITS = {"Length": (0.05, 0.90), "Diameter": (0.05, 0.70), "Height": (0.01, 1.20),
          "Whole_weight": (0.002, 3.0), "Shucked_weight": (0.001, 1.6),
          "Viscera_weight": (0.0005, 0.8), "Shell_weight": (0.002, 1.1)}


@st.cache_resource
def load_model():
    return joblib.load("abalone_model.joblib")


def add_features(frame):
    out = frame.copy()
    out["Volume"] = out["Length"] * out["Diameter"] * out["Height"]
    out["Shell_ratio"] = out["Shell_weight"] / out["Whole_weight"]
    out["Meat_ratio"] = out["Shucked_weight"] / out["Whole_weight"]
    return out


def predict(frame):
    model = load_model()
    rings = model.predict(add_features(frame[FEATURES]))
    result = frame[FEATURES].copy()
    result["Predicted_rings"] = rings.round(1)
    result["Predicted_age_years"] = (rings + 1.5).round(1)
    return result


def validate(frame):
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        return "Missing columns: " + ", ".join(missing)
    data = frame[FEATURES]
    if data.isna().any().any():
        return "The file has empty cells in the required columns."
    bad_sex = ~data["Sex"].isin(["M", "F", "I"])
    if bad_sex.any():
        return "Sex must be M, F or I (problem in row(s): %s)." % list(data.index[bad_sex][:5] + 2)
    for col in FEATURES[1:]:
        if not pd.to_numeric(data[col], errors="coerce").notna().all():
            return "Column %s must contain numbers only." % col
    if (data["Height"].astype(float) <= 0).any() or (data["Whole_weight"].astype(float) <= 0).any():
        return "Height and Whole_weight must be greater than zero."
    return None


st.set_page_config(page_title="Abalone age predictor", page_icon="🐚")
st.title("🐚 Abalone age predictor")
st.caption("Predicts the number of shell rings (age in years = rings + 1.5). "
           "Measurements use the dataset's scaled units (original value divided by 200).")

tab1, tab2 = st.tabs(["Single abalone", "Upload Excel / CSV"])

with tab1:
    sex = st.selectbox("Sex (M = male, F = female, I = infant)", ["M", "F", "I"])
    cols = st.columns(2)
    values = {}
    for i, name in enumerate(FEATURES[1:]):
        lo, hi = LIMITS[name]
        values[name] = cols[i % 2].number_input(name, min_value=lo, max_value=hi,
                                                value=round((lo + hi) / 4, 3), format="%.4f")
    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{"Sex": sex, **values}])
        err = validate(row)
        if err:
            st.error(err)
        else:
            res = predict(row).iloc[0]
            st.success("Predicted rings: %.1f  |  estimated age: %.1f years"
                       % (res["Predicted_rings"], res["Predicted_age_years"]))

with tab2:
    st.write("Upload a file with these columns: " + ", ".join(FEATURES))
    template = pd.DataFrame([{"Sex": "M", "Length": 0.455, "Diameter": 0.365, "Height": 0.095,
                              "Whole_weight": 0.514, "Shucked_weight": 0.2245,
                              "Viscera_weight": 0.101, "Shell_weight": 0.15}])
    buf = io.BytesIO()
    template.to_excel(buf, index=False)
    st.download_button("Download template (Excel)", buf.getvalue(), "abalone_template.xlsx")
    up = st.file_uploader("Excel or CSV file", type=["xlsx", "xls", "csv"])
    if up is not None:
        try:
            data = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        except Exception as e:
            st.error("Could not read the file: %s" % e)
        else:
            err = validate(data)
            if err:
                st.error(err)
            else:
                data[FEATURES[1:]] = data[FEATURES[1:]].astype(float)
                out = predict(data)
                st.dataframe(out)
                st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                                   "abalone_predictions.csv", "text/csv")
