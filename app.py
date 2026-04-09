import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

twilio_sid   = os.getenv("TWILIO_SID")
twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_from  = os.getenv("TWILIO_FROM")
twilio_to    = os.getenv("TWILIO_TO")

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HeatGuard AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "saved_models")
PLOT_DIR   = os.path.join(BASE_DIR, "plots")

LABEL_NAMES = {0: "Normal", 1: "Moderate", 2: "High"}
LABEL_COLORS = {0: "#27AE60", 1: "#F39C12", 2: "#E74C3C"}
LABEL_ICONS  = {0: "✅", 1: "⚠️", 2: "🚨"}

MODEL_DISPLAY = {
    "custom_nn"     : " Custom Neural Network",
    "gat"           : " GAT Neural Network",
    "random_forest" : " Random Forest",
    "gradient_boost": " Gradient Boosting",
    "svm"           : " SVM (RBF)",
    "logistic_reg"  : " Logistic Regression",
}

FEATURES = [
    "body_temp", "ambient_temp", "humidity", "heart_rate",
    "skin_resistance", "resp_rate", "movement",
    "avg_sensor_temp", "sensor_spread",
    "temp_humidity_index", "heat_index",
    "hr_temp_product", "skin_resistance_normalized",
    "body_amb_diff", "iaq", "lux", "sound"
]

COLORS = {
    "custom_nn"     : "#7F77DD",
    "gat"     : "#7F77DD",
    "random_forest" : "#1D9E75",
    "gradient_boost": "#EF9F27",
    "svm"           : "#D85A30",
    "logistic_reg"  : "#378ADD",
}

st.markdown("""
<style>
    .risk-card {
        border-radius: 14px;
        padding: 22px 28px;
        margin: 10px 0;
        font-size: 1.15em;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .risk-normal   { background: #d4edda; color: #155724; border: 2px solid #27AE60; }
    .risk-moderate { background: #fff3cd; color: #856404; border: 2px solid #F39C12; }
    .risk-high     { background: #f8d7da; color: #721c24; border: 2px solid #E74C3C; }
    .metric-box {
        background: #f8f9fc;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        border: 1px solid #e1e4ea;
    }
    .metric-val  { font-size: 1.7em; font-weight: 700; color: #2c3e50; }
    .metric-lbl  { font-size: 0.8em; color: #6c757d; margin-top: 2px; }
    .alert-banner {
        background: #E74C3C; color: white;
        border-radius: 10px; padding: 14px 20px;
        font-weight: 700; font-size: 1.05em;
        animation: pulse 1s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.8} }
    .section-header {
        font-size: 1.15em; font-weight: 700;
        color: #2c3e50; margin: 18px 0 10px 0;
        padding-bottom: 4px; border-bottom: 2px solid #eee;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; border-radius: 8px 8px 0 0;
        font-weight: 600; font-size: 0.95em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models, features = {}, {}
    for target in ["heat_stress_label", "dehydration_label"]:
        models[target] = {}
        for name in MODEL_DISPLAY:
            path = os.path.join(MODEL_DIR, f"{target}_{name}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    models[target][name] = pickle.load(f)
        feat_path = os.path.join(MODEL_DIR, f"{target}_features.json")
        if os.path.exists(feat_path):
            with open(feat_path) as f:
                features[target] = json.load(f)
    return models, features

@st.cache_data
def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def compute_features(inputs):
    bt  = inputs["body_temp"]
    at  = inputs["ambient_temp"]
    hum = inputs["humidity"]
    hr  = inputs["heart_rate"]
    sr  = inputs["skin_resistance"]
    rr  = inputs.get("resp_rate", 18)
    mv  = inputs.get("movement", 1)

    thi = bt + 0.33 * (hum / 100 * 6.105 * np.exp(17.27 * at / (at + 237.3))) - 4.0
    hi  = (-8.78 + 1.611 * at + 2.339 * hum - 0.1461 * at * hum
           - 0.0123 * at**2 - 0.0164 * hum**2
           + 0.00221 * at**2 * hum + 0.000725 * at * hum**2
           - 3.58e-6 * at**2 * hum**2)

    return {
        "body_temp": bt, "ambient_temp": at, "humidity": hum,
        "heart_rate": hr, "skin_resistance": sr,
        "resp_rate": rr, "movement": mv,
        "avg_sensor_temp": bt, "sensor_spread": 0.2,
        "temp_humidity_index": thi, "heat_index": hi,
        "hr_temp_product": hr * bt / 100.0,
        "skin_resistance_normalized": sr / 500.0,   
        "body_amb_diff": bt - at,
        "iaq": 0.0, "lux": 0.0, "sound": 0.0,
    }

def predict_risk(model, feat_row, feature_names):
    X     = np.array([[feat_row.get(f, 0.0) for f in feature_names]], dtype=np.float32)
    proba = model.predict_proba(X)[0]
    
    thresholds = getattr(model, '_heatguard_thresholds', None)
    if thresholds is not None and len(thresholds) == len(proba):
        adj  = proba / (np.array(thresholds) + 1e-9)
        pred = int(adj.argmax())
    else:
        pred = int(proba.argmax())
    return pred, proba

def make_proba_chart(proba_h, proba_d):
    """Always display all 3 class bars — pads if model outputs < 3 classes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    cats       = ["Normal", "Moderate", "High"]
    bar_colors = ["#27AE60", "#F39C12", "#E74C3C"]

    def pad3(p):
        p = np.array(p, dtype=float)
        out = np.zeros(3)
        out[:len(p)] = p[:3]
        return out

    for ax, proba_raw, title in [(ax1, proba_h, "Heat Stress"),
                                  (ax2, proba_d, "Dehydration")]:
        proba = pad3(proba_raw)
        bars  = ax.bar(cats, proba * 100, color=bar_colors, width=0.55, edgecolor="white", lw=1.5)
        for bar, v in zip(bars, proba):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
        best = int(np.argmax(proba))
        bars[best].set_edgecolor("#2c3e50"); bars[best].set_linewidth(2.5)
        ax.set_ylim(0, 120); ax.set_ylabel("Probability (%)")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.grid(True, axis="y", alpha=0.25); ax.set_axisbelow(True)
    plt.tight_layout()
    return fig

def make_gauge(value, max_val, label, color):
    fig, ax = plt.subplots(figsize=(2.5, 2.5), subplot_kw=dict(aspect='equal'))
    theta = np.linspace(0, np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), lw=10, color='#ecf0f1', solid_capstyle='round')
    fill = np.pi * value / max_val
    ax.plot(np.cos(np.linspace(0, fill, 200)), np.sin(np.linspace(0, fill, 200)),
            lw=10, color=color, solid_capstyle='round')
    ax.text(0, -0.2, f"{value:.1f}", ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0, -0.55, label, ha='center', va='center', fontsize=8, color='#666')
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.8, 1.3); ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def make_comparison_chart(metrics):
    if not metrics: return None
    targets     = list(metrics.keys())
    model_names = list(next(iter(metrics.values())).keys())
    metric_keys = [("accuracy","Accuracy"), ("f1_weighted","F1 Score"), ("roc_auc","ROC AUC")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(model_names)); w = 0.35

    for ax, (mkey, mlabel) in zip(axes, metric_keys):
        for ti, target in enumerate(targets):
            vals   = [metrics[target].get(n, {}).get(mkey, 0) for n in model_names]
            offset = (ti - 0.5) * w
            cols   = [COLORS.get(n, "#888") for n in model_names]
            bars   = ax.bar(x + offset, vals, w,
                            label=target.replace("_label","").replace("_"," ").title(),
                            color=cols if ti == 0 else [c+"99" for c in cols],
                            edgecolor="white", lw=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                        f"{v:.3f}", ha='center', fontsize=6.5, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_","\n").title() for n in model_names], fontsize=8)
        ax.set_title(mlabel, fontweight='bold'); ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.2); ax.set_axisbelow(True)
        ax.axhline(0.95, color='#E74C3C', linestyle='--', lw=1.2, alpha=0.6)
    plt.suptitle("Model Performance Comparison — HeatGuard AI",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig

def make_cv_chart(metrics):
    if not metrics: return None
    targets     = list(metrics.keys())
    model_names = list(next(iter(metrics.values())).keys())
    fig, axes   = plt.subplots(1, len(targets), figsize=(9*len(targets), 5))
    if len(targets) == 1: axes = [axes]
    for ax, target in zip(axes, targets):
        means = [metrics[target].get(n, {}).get('cv_mean', 0) for n in model_names]
        stds  = [metrics[target].get(n, {}).get('cv_std',  0) for n in model_names]
        cols  = [COLORS.get(n,'#888') for n in model_names]
        bars  = ax.bar(model_names, means, color=cols, edgecolor='white', lw=0.8, alpha=0.92)
        ax.errorbar(model_names, means, yerr=stds, fmt='none', color='#2C2C2A',
                    capsize=6, lw=2, capthick=2)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.005,
                    f"{m:.3f}\n±{s:.3f}", ha='center', fontsize=8)
        ax.set_ylim(0, 1.15); ax.set_ylabel('CV Accuracy')
        ax.set_xticklabels([n.replace('_','\n').title() for n in model_names], fontsize=9)
        ax.set_title(f"5-Fold CV — {target.replace('_label','').replace('_',' ').title()}",
                     fontsize=12, fontweight='bold')
        ax.axhline(0.95, color='#E74C3C', linestyle='--', lw=1.5)
        ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    return fig

def main():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/temperature.png", width=64)
        st.title("HeatGuard AI")
        st.caption("Early Heat Stress & Dehydration Detection")
        st.divider()

        st.subheader(" Model Selection")
        model_sel = st.selectbox("Prediction Model", list(MODEL_DISPLAY.keys()),
                                 format_func=lambda k: MODEL_DISPLAY[k])
        st.markdown("""
            <style>
            div[data-baseweb="select"] > div {
                cursor: pointer;
            }
            </style>
        """, unsafe_allow_html=True)
        st.divider()

        st.subheader(" About")
        st.markdown("""
        **3-class risk system:**
        - ✅ **Normal** — Safe zone
        - ⚠️ **Moderate** — Monitor closely
        - 🚨 **High** — Immediate action

        **Models:**
        -  Custom Deep NN (NumPy)
        -  Random Forest
        -  Gradient Boosting
        -  SVM (RBF)
        -  Logistic Regression
        """)

    models, features = load_models()
    metrics          = load_metrics()
    models_loaded    = any(models.get(t) for t in models)

    if not models_loaded:
        st.error(" No trained models found. Run `python train_models.py` first.")
        st.code("python train_models.py", language="bash")
        st.stop()

    tabs = st.tabs([" Live Prediction", "📊 Model Comparison", "📈 Cross-Validation", "ℹ️ About"])

    with tabs[0]:
        def send_alert(msg):
            # account_sid = twilio_sid
            # auth_token = twilio_token
            # client = Client(account_sid, auth_token)

            # message = client.messages.create(
            #     from_ = twilio_from,
            #     body=msg,
            #     to = twilio_to
            # )
            # print(message.sid)
            return
        st.header(" Real-Time Risk Prediction")

        col_inp, col_res = st.columns([1, 1.4], gap="large")

        with col_inp:
            st.markdown('<div class="section-header"> Sensor Inputs</div>', unsafe_allow_html=True)

            bt  = st.slider(" Body Temperature (°C)", 35.0, 42.0, 37.5, 0.1)
            at  = st.slider(" Ambient Temperature (°C)", 15.0, 50.0, 28.0, 0.5)
            hr  = st.slider(" Heart Rate (bpm)", 40, 200, 90)
            hum = st.slider(" Humidity (%)", 10.0, 100.0, 60.0, 1.0)
            sr  = st.slider(" Skin Resistance (Ω)", 20.0, 450.0, 100.0, 5.0)
            rr  = st.slider(" Respiration Rate (bpm)", 10, 40, 18)
            mv  = st.slider(" Movement Level", 0, 5, 1)

            st.markdown('<div class="section-header"> Computed Indices</div>', unsafe_allow_html=True)
            feat = compute_features({"body_temp": bt, "ambient_temp": at, "humidity": hum,
                                     "heart_rate": hr, "skin_resistance": sr,
                                     "resp_rate": rr, "movement": mv})
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Temp-Humidity Index", f"{feat['temp_humidity_index']:.2f} °C")
            with c2:
                st.metric("Heat Index", f"{feat['heat_index']:.2f} °C")
            c3, c4 = st.columns(2)
            with c3:
                st.metric("Body-Ambient ΔT", f"{feat['body_amb_diff']:.2f} °C")
            with c4:
                st.metric("HR * Temp Product", f"{feat['hr_temp_product']:.1f}")

        with col_res:
            st.markdown('<div class="section-header"> Prediction Results</div>', unsafe_allow_html=True)

            heat_pred, dehyd_pred = None, None
            proba_h, proba_d      = None, None

            for target in ["heat_stress_label", "dehydration_label"]:
                if model_sel in models.get(target, {}):
                    feats = features.get(target, FEATURES)
                    p, pr = predict_risk(models[target][model_sel], feat, feats)
                    if target == "heat_stress_label":
                        heat_pred, proba_h = p, pr
                    else:
                        dehyd_pred, proba_d = p, pr

            if heat_pred is not None:
                
                for lbl, pred in [(" Heat Stress Risk", heat_pred),
                                   (" Dehydration Risk", dehyd_pred)]:
                    level_cls = {0: "normal", 1: "moderate", 2: "high"}[pred]
                    st.markdown(
                        f'<div class="risk-card risk-{level_cls}">'
                        f'{LABEL_ICONS[pred]} {lbl}: <strong>{LABEL_NAMES[pred]}</strong>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                overall = max(heat_pred, dehyd_pred)
                st.markdown("---")
                if overall == 2:
                    st.markdown(
                        '<div class="alert-banner">🚨 HIGH RISK — Immediate attention required!</div>',
                        unsafe_allow_html=True)
                    message = (
                        f"⚠️ Health Alert ⚠️\n\n"
                        f"Risk Level: HIGH\n\n"
                        f"Heat Risk: {proba_h[overall] * 100:.2f}\n\n"
                        f"Dehydration Risk: {proba_d[overall] * 100:.2f}\n\n"
                        f"Please take immediate precautions:\n"
                        f"- Stay hydrated 💧\n"
                        f"- Avoid direct sunlight ☀️\n"
                        f"- Take rest if feeling unwell 🛑\n\n"
                        f"Stay safe!"
                    )
                    send_alert(message)
                elif overall == 1:
                    st.warning(" **MODERATE RISK** — Please monitor closely and consider rest + hydration.")
                    message = (
                        f"⚠️ Health Alert ⚠️\n\n"
                        f"Risk Level: MODERATE\n\n"
                        f"Heat Risk: {proba_h[overall] * 100:.2f}\n\n"
                        f"Dehydration Risk: {proba_d[overall] * 100:.2f}\n\n"
                        f"Please take immediate precautions:\n"
                        f"- Stay hydrated 💧\n"
                        f"- Avoid direct sunlight ☀️\n"
                        f"- Take rest if feeling unwell 🛑\n\n"
                        f"Stay safe!"
                    )
                    send_alert(message)

                else:
                    st.success(" **NORMAL** — All vitals within safe ranges.")

                st.markdown('<div class="section-header">📊 Probability Breakdown</div>',
                            unsafe_allow_html=True)
                st.pyplot(make_proba_chart(proba_h, proba_d))

                
                st.markdown('<div class="section-header"> Confidence Metrics</div>',
                            unsafe_allow_html=True)
                cc1, cc2, cc3, cc4 = st.columns(4)
                with cc1:
                    st.metric("Heat Confidence", f"{proba_h.max()*100:.1f}%")
                with cc2:
                    st.metric("Dehyd Confidence", f"{proba_d.max()*100:.1f}%")
                with cc3:
                    st.metric("Heat Class", LABEL_NAMES[heat_pred])
                with cc4:
                    st.metric("Dehyd Class", LABEL_NAMES[dehyd_pred])

            else:
                st.error(f"Model `{model_sel}` not found. Run training first.")

    
    with tabs[1]:
        st.header("📊 Model Performance Comparison")

        if not metrics:
            st.warning("No metrics found. Run `python train_models.py` first.")
        else:
            
            st.markdown('<div class="section-header">Performance Table</div>', unsafe_allow_html=True)
            rows = []
            for target, mdict in metrics.items():
                for model, m in mdict.items():
                    rows.append({
                        "Target"   : target.replace("_label","").replace("_"," ").title(),
                        "Model"    : MODEL_DISPLAY.get(model, model),
                        "Accuracy" : round(m.get("accuracy", 0), 4),
                        "F1 Score" : round(m.get("f1_weighted", 0), 4),
                        "ROC AUC"  : round(m.get("roc_auc", 0), 4),
                        "CV Mean"  : round(m.get("cv_mean", 0), 4),
                        "CV Std"   : round(m.get("cv_std", 0), 4),
                    })
            df_table = pd.DataFrame(rows)

            def style_acc(v):
                if isinstance(v, float):
                    if v >= 0.95:  return 'background-color: #d4edda; color: #155724; font-weight:bold'
                    if v >= 0.85:  return 'background-color: #fff3cd; color: #856404'
                    if v < 0.75:   return 'background-color: #f8d7da; color: #721c24'
                return ''

            st.dataframe(
                df_table.style.map(style_acc, subset=["Accuracy","F1 Score","ROC AUC","CV Mean"]),
                use_container_width=True, hide_index=True
            )

            
            best_row = df_table.loc[df_table["Accuracy"].idxmax()]
            st.success(f" Best Model: **{best_row['Model'].strip()}** on **{best_row['Target']}** "
                       f"— Accuracy: **{best_row['Accuracy']:.4f}**")

            st.markdown("---")
            st.markdown('<div class="section-header">Comparison Charts</div>', unsafe_allow_html=True)

            
            comp_path = os.path.join(PLOT_DIR, "model_comparison.png")
            if os.path.exists(comp_path):
                st.image(comp_path, use_container_width=True)
            else:
                fig = make_comparison_chart(metrics)
                if fig: st.pyplot(fig)

            
            for target in metrics:
                title = target.replace("_label","").replace("_"," ").title()
                roc_path  = os.path.join(PLOT_DIR, f"roc_{target}.png")
                conf_path = os.path.join(PLOT_DIR, f"confusion_{target}.png")
                f1_path   = os.path.join(PLOT_DIR, f"per_class_f1_all.png")
                st.markdown(f'<div class="section-header">{title} — Detailed Plots</div>',
                            unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    if os.path.exists(roc_path):
                        st.image(roc_path, caption="ROC Curves", use_container_width=True)
                with c2:
                    if os.path.exists(conf_path):
                        st.image(conf_path, caption="Confusion Matrices", use_container_width=True)
            
            f1_all = os.path.join(PLOT_DIR, "per_class_f1_all.png")
            if os.path.exists(f1_all):
                st.markdown('<div class="section-header">Per-Class F1 Score Breakdown</div>',
                            unsafe_allow_html=True)
                st.image(f1_all, use_container_width=True)

    
    with tabs[2]:
        st.header("📈 Cross-Validation Analysis")

        if not metrics:
            st.warning("No metrics found. Run `python train_models.py` first.")
        else:
            cv_path = os.path.join(PLOT_DIR, "cross_validation.png")
            if os.path.exists(cv_path):
                st.image(cv_path, use_container_width=True)
            else:
                fig = make_cv_chart(metrics)
                if fig: st.pyplot(fig)

            st.markdown("---")
            st.markdown('<div class="section-header">CV Scores Detail</div>', unsafe_allow_html=True)
            for target, mdict in metrics.items():
                title = target.replace("_label","").replace("_"," ").title()
                st.subheader(title)
                cv_rows = [{"Model": MODEL_DISPLAY.get(m, m),
                            "CV Mean": f"{v.get('cv_mean',0):.4f}",
                            "CV Std":  f"± {v.get('cv_std',0):.4f}",
                            "Status":  " Excellent" if v.get('cv_mean',0) >= 0.95
                                       else (" Good" if v.get('cv_mean',0) >= 0.85
                                             else " Needs Improvement")}
                           for m, v in mdict.items()]
                st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.header("ℹ️ About HeatGuard AI")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
###  Classification System

| Class | Risk Level | Action |
|-------|-----------|--------|
| 0 |  Normal | Routine monitoring |
| 1 |  Moderate | Rest + hydrate, monitor |
| 2 |  High | Immediate medical help |

###  Input Features
- Body & Ambient Temperature
- Heart Rate & Skin Resistance
- Humidity & Respiration Rate
- Movement Level
- Derived: Heat Index, THI, ΔT
            """)
        with col2:
            st.markdown("""
###  Custom Neural Network (NumPy)
Built entirely from scratch — no sklearn MLPClassifier:
- **Architecture:** 4 layers (128→64→32→3)
- **Activations:** ReLU + Softmax
- **Optimiser:** Adam with LR decay
- **Regularisation:** L2 + Dropout + Batch Norm
- **Training:** Early stopping (patience=20)

### 📲 WhatsApp Alert (Twilio)
Sends a whatsapp alert to the configured mobile number when the prediction results in **Moderate (1)** or **High (2)** risk.
            """)

if __name__ == "__main__":
    main()