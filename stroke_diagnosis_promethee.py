import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(layout="wide")
st.title("PROMETHEE Stroke Diagnostic Tool")

# --- Default Data ---
devices = ["CT Scan", "MRI", "Transcranial Doppler", "EEG", "CT Angiography", "VIPS Visor"]
criteria_names = ["Sensitivity", "Specificity", "Accuracy", "Radiation", "Cost", "Availability", "Time", "Portability", "3D Imaging"]
criteria_types = ["max", "max", "max", "min", "min", "max", "min", "max", "max"]
weights = np.array([0.92, 0.92, 0.92, 0.08, 0.75, 0.50, 0.50, 0.25, 0.25])
symbol_map = {"low": 3.0, "high": 9.0, "no": 1.0, "yes": 10.0}

raw_data = [
    [26.0, 98.0, 54.0, 1.7, 0.50, "low", 1.0, "no", "yes"],
    [83.0, 97.0, 89.0, 0.0, 0.92, "low", 60.0, "no", "yes"],
    [79.1, 94.3, 89.4, 0.0, 0.08, "high", 37.5, "yes", "yes"],
    [80.0, 93.0, 91.3, 0.0, 0.25, "high", 30.0, "yes", "no"],
    [83.2, 95.0, 94.0, 1.9, 0.50, "low", 1.0, "no", "yes"],
    [93.0, 92.0, 93.0, 0.0, 0.25, "low", 0.5, "yes", "no"]
]

# --- Functions ---
def linear_pref(d): return min(1, d / 3)

def convert_data(raw, names):
    matrix = []
    for i, row in enumerate(raw):
        new_row = row[:]
        new_row[5] = symbol_map.get(row[5], 0)
        new_row[7] = symbol_map.get(row[7], 0)
        new_row[8] = 10.0 if names[i] == "VIPS Visor" else 8.0 if names[i] == "CT Angiography" else symbol_map.get(row[8], 0)
        matrix.append(new_row)
    return np.array(matrix)

def run_promethee(matrix, weights, types):
    n, m = matrix.shape
    phi_plus, phi_minus = np.zeros(n), np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            pref_ij = sum(weights[k] * linear_pref(max(0, (matrix[i][k] - matrix[j][k]) if types[k]=="max" else (matrix[j][k] - matrix[i][k])))
                          for k in range(m))
            pref_ji = sum(weights[k] * linear_pref(max(0, (matrix[j][k] - matrix[i][k]) if types[k]=="max" else (matrix[i][k] - matrix[j][k])))
                          for k in range(m))
            phi_plus[i] += pref_ij
            phi_minus[i] += pref_ji
    phi_plus /= (n - 1)
    phi_minus /= (n - 1)
    return phi_plus - phi_minus, phi_plus, phi_minus

def show_chart(names, phi_plus, phi_minus):
    phi = phi_plus - phi_minus
    sorted_indices = np.argsort(phi)[::-1]
    names_sorted = [f"{names[i]} (#{sorted_indices.tolist().index(i)+1})" for i in sorted_indices]
    phi_plus_sorted = phi_plus[sorted_indices]
    phi_minus_sorted = phi_minus[sorted_indices]
    phi_sorted = phi[sorted_indices]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, phi_plus_sorted, width, label='Phi+', color='#1f77b4')
    ax.bar(x + width/2, phi_minus_sorted, width, label='Phi−', color='#ff7f0e')

    for i in range(len(x)):
        ax.text(x[i], max(phi_plus_sorted[i], phi_minus_sorted[i]) + 0.05,
                f"Φ = {phi_sorted[i]:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(names_sorted, rotation=45, ha='right')
    ax.set_ylabel("Flow Value")
    ax.set_title("PROMETHEE Flows Ranked by Phi (Φ)")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)
    st.markdown("""
    **Interpretation**:  
    This bar chart shows the PROMETHEE positive (Phi⁺) and negative (Phi⁻) preference flows for each diagnostic device.  
    The net flow (Φ) determines the final ranking. A higher Phi⁺ and lower Phi⁻ indicate a more preferred device.
    """)

def show_pie_chart(names, phi):
    phi_shifted = phi - np.min(phi) + 1
    phi_normalized = phi_shifted / np.sum(phi_shifted)

    sorted_indices = np.argsort(phi)[::-1]
    phi_sorted = phi_normalized[sorted_indices]
    names_sorted = [f"{names[i]} (#{sorted_indices.tolist().index(i)+1})" for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        phi_sorted,
        labels=names_sorted,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.85,
        textprops={'fontsize': 9}
    )

    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig.gca().add_artist(centre_circle)

    ax.set_title("Ranking Distribution by Phi (Adjusted Donut Chart)", fontsize=12)
    st.pyplot(fig)
    st.markdown("""
    **Interpretation**:  
    This donut chart represents the proportional ranking of devices based on their net Phi (Φ) values.  
    Devices with higher Phi values take larger portions, showing stronger overall performance in the PROMETHEE analysis.
    """)

def show_results(names, phi, phi_plus, phi_minus):
    df = pd.DataFrame({
        "Device": names,
        "Phi+": phi_plus,
        "Phi−": phi_minus,
        "Phi": phi,
    })
    df["Rank"] = df["Phi"].rank(ascending=False, method='min').astype(int)
    df = df.sort_values("Rank").reset_index(drop=True)
    df = df[["Rank", "Device", "Phi", "Phi+", "Phi−"]]

    def highlight_best(val, column):
        if column == "Device" and val == "VIPS Visor":
            return 'background-color: #388e3c; color: white; font-weight: bold'
        elif column == "Rank" and val == 1:
            return 'background-color: #388e3c; color: white; font-weight: bold'
        else:
            return ''

    styled_df = df.style.format({"Phi": "{:.3f}", "Phi+": "{:.3f}", "Phi−": "{:.3f}"})
    for col in df.columns:
        styled_df = styled_df.applymap(lambda val: highlight_best(val, col), subset=[col])
    styled_df = styled_df.set_properties(**{
        'font-size': '14px',
        'text-align': 'center',
        'padding': '6px'
    })

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### PROMETHEE Results")
        st.dataframe(styled_df, hide_index=True)

    show_chart(df["Device"].values, df["Phi+"].values, df["Phi−"].values)
    show_pie_chart(df["Device"].values, df["Phi"].values)

# --- Interface ---
mode = st.radio("Select Mode", ["Use Thesis Data", "Enter Custom Data"])

if mode == "Use Thesis Data":
    st.subheader("Raw Data (Thesis)")
    st.dataframe(pd.DataFrame(raw_data, columns=criteria_names, index=devices))
    if st.button("Run PROMETHEE on Original Data"):
        data_matrix = convert_data(raw_data, devices)
        phi, phi_plus, phi_minus = run_promethee(data_matrix, weights, criteria_types)
        show_results(devices, phi, phi_plus, phi_minus)

else:
    st.subheader("Step 1: Enter Criteria")
    num_criteria = st.number_input("Number of Criteria", 2, 20, 3)
    c_names, c_types, c_weights = [], [], []

    for i in range(num_criteria):
        col1, col2, col3 = st.columns(3)
        with col1:
            default_label = f"Criterion {i+1}"
            name = st.text_input(default_label, key=f"cname_{i}")
            c_names.append(name)
        with col2:
            c_types.append(st.selectbox("Type", ["min", "max"], key=f"ctype_{i}"))
        with col3:
            c_weights.append(st.number_input("Weight", 0.0, 1.0, 0.5, 0.01, key=f"cweight_{i}"))

    st.subheader("Step 2: Enter Devices")
    num_devices = st.number_input("Number of Devices", 2, 20, 3)
    d_names, d_values = [], []

    for i in range(num_devices):
        with st.expander(f"Device {i+1}"):
            d_names.append(st.text_input("Device Name", key=f"dname_{i}"))
            row = []
            for j in range(num_criteria):
                criterion_label = c_names[j] if c_names[j].strip() else f"Criterion {j+1}"
                col1, col2 = st.columns([1, 2])
                with col1:
                    opt = st.selectbox(criterion_label, ["", "yes", "no", "high", "low"], key=f"opt_{i}_{j}")
                with col2:
                    val = st.text_input("Or number", key=f"val_{i}_{j}")
                row.append(float(val) if val else symbol_map.get(opt.lower(), 0))
            d_values.append(row)

    if st.button("Run PROMETHEE on Custom Data"):
        st.subheader("Your Entered Data")
        data = np.array(d_values)
        st.dataframe(pd.DataFrame(data, columns=c_names, index=d_names))
        phi, phi_plus, phi_minus = run_promethee(data, np.array(c_weights), c_types)
        show_results(d_names, phi, phi_plus, phi_minus)

# --- Fixed Footer ---
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    padding: 10px 0;
    background: #0e1117;
    text-align: center;
    font-size: 13px;
    color: #cccccc;
    border-top: 1px solid #222;
    z-index: 9999;
}
</style>
<div class="footer">
© 2025 Muntadher Taher Alhatab. This tool was developed as part of the Master's Thesis project on stroke diagnosis evaluation using the PROMETHEE method. All rights reserved.
</div>
""", unsafe_allow_html=True)
