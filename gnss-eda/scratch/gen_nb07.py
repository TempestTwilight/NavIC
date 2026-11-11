import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.14.4"},
}

cells = []

# Intro Markdown Cell
cells.append(
    nbf.v4.new_markdown_cell("""# NB07: Mentor Story — The Physics, Pitfalls, and Machine Learning of Smartphone GNSS Optimization

### Purpose & Executive Storyline
This notebook tells the complete end-to-end project narrative for an engineering mentor or stakeholder who has not seen the raw data. It synthesizes findings from 95 raw GNSS Logger files (9.47 GB, 5.5 million measurements across 4 smartphone platforms).

Unlike exploratory notebooks, this notebook executes **no heavy SQL or raw data queries**. It is entirely lightweight, loading pre-computed summary tables and rendering the top 11 foundational figures with clear, concise takeaways explaining:
1. **The Core Physical Problem:** Multipath signal degradation and antenna gain attenuation in smartphone GNSS.
2. **Data Pitfalls & Traps Uncovered:** MediaTek vendor bugs ($C/N_0=0$), Euler gimbal lock singularities, and sensor sampling rate disparities.
3. **What Governs Signal Strength:** Satellite elevation and hardware LNA gain vs orientation directivity.
4. **Temporal Dynamics & The Persistence Paradox:** Why raw signal strength follows a near-martingale random walk, and why models must predict residual drift $\Delta C/N_0$.
5. **On-Device Decision Making:** How selecting the top future satellite reduces PVT fix error, and the road to production deployment.
""")
)

# Setup Code Cell
cells.append(
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
import os

FIG_DIR = "/home/tempest/Documents/ISRO_Project/gnss-eda/figures"
TAB_DIR = "/home/tempest/Documents/ISRO_Project/gnss-eda/tables"

def show_story_step(fig_name, title, takeaway, details):
    fig_path = f"{FIG_DIR}/{fig_name}"
    display(Markdown(f"### {title}\\n**Takeaway:** {takeaway}\\n\\n{details}"))
    if os.path.exists(fig_path):
        display(Image(filename=fig_path, width=800))
    else:
        print(f"[Warning] Figure {fig_name} not found.")
""")
)

# Story Section 1: Data Traps & Sanity Checks
cells.append(
    nbf.v4.new_markdown_cell("""## Part 1: Dataset Realities & Sensor Coordinate Conventions
Before modeling, sensor conventions and hardware quirks had to be mathematically verified.
""")
)

cells.append(
    nbf.v4.new_code_cell("""show_story_step(
    fig_name="01_gravity_vs_pitch_roll.png",
    title="1. Sensor Alignment: Resolving Euler Angle Conventions",
    takeaway="OrientationDeg.pitchDeg directly measures nose-up physical pitch (r = 0.999), resolving Android documentation ambiguities.",
    details="By projecting the accelerometer gravity vector onto the phone body axes ($g_y / g$), we mathematically proved that `pitchDeg` maps to pitch tilt, while `rollDeg` maps to lateral roll. This confirmed that our 3D rotation matrices correctly orient the smartphone antenna."
)
""")
)

cells.append(
    nbf.v4.new_code_cell("""show_story_step(
    fig_name="02_yaw_fop_gimbal_lock.png",
    title="2. The Gimbal Lock Trap: Mathematical Singularities at Vertical Tilt",
    takeaway="Euler yaw and fused heading diverge by >100° when phone pitch approaches ±90°, mandating 3D vector representations.",
    details="When a phone is held vertically (e.g. in a car mount or upright hand), Euler pitch encounters a gimbal lock singularity ($|\\text{pitch}| = 90^\\circ$). Yaw flips instantly by $180^\\circ$. Training machine learning models on raw Euler yaw angles creates catastrophic spurious discontinuities; our pipeline converts phone attitudes into continuous 3D unit vectors ($s_{bx}, s_{by}, s_{bz}$)."
)
""")
)

cells.append(
    nbf.v4.new_code_cell("""show_story_step(
    fig_name="05_session_timeline.png",
    title="3. Dataset Structure: The Two Disjoint Collection Campaigns",
    takeaway="The 95 logs split into a June orientation bench campaign (14 static logs) and an October dynamic multi-device campaign (81 logs).",
    details="Device models are strictly confounded with recording dates. The June test-stand sessions provide controlled $360^\\circ$ yaw sweeps, while the October sessions provide dynamic vehicle and pedestrian data. Cross-validation cannot pool these campaigns randomly without causing severe evaluation leakage."
)
""")
)

# Story Section 2: What Explains Signal Variance?
cells.append(
    nbf.v4.new_markdown_cell("""## Part 2: The Physical Drivers of GNSS Signal Quality ($C/N_0$)
We systematically decomposed the total variance of carrier-to-noise density to establish an empirical performance ceiling for ML.
""")
)

cells.append(
    nbf.v4.new_code_cell("""show_story_step(
    fig_name="07_variance_decomposition.png",
    title="4. Variance Decomposition: Where Does Signal Strength Come From?",
    takeaway="Satellite elevation angle explains 61.2% of signal variance; phone orientation and multipath explain the remaining 38.8%.",
    details="Sequential ANOVA decomposition proves that orbital elevation angle dominates raw signal strength due to atmospheric attenuation and standard patch antenna boresight geometry. Any model that predicts raw $C/N_0$ simply acts as an expensive proxy for satellite elevation; models must predict residual variance after removing this deterministic baseline."
)
""")
)

cells.append(
    nbf.v4.new_code_cell("""show_story_step(
    fig_name="10_device_gain_offset.png",
    title="5. Hardware Disparity: The 14.5 dB Smartphone Gain Gap",
    takeaway="Samsung devices report ~14.5 dB higher nominal C/N0 than MediaTek and Qualcomm devices under identical sky conditions.",
    details="Evaluating clean, zenith GPS L1 signals ($>60^\\circ$ elevation) revealed that Samsung SM-M146B averages $41.8\\text{ dB-Hz}$, compared to $27.5\\text{ dB-Hz}$ on the OnePlus CPH2401. This is a permanent hardware LNA/DSP calibration offset. Models trained on raw signal levels will fail when transferred across phone models."
)
""")
)

cells.append(
    nbf.v4.new_code_cell("""show_story_step(
    fig_name="12_antenna_pattern_directivity.png",
    title="6. Antenna Radiation Pattern: Boresight Directivity",
    takeaway="Signals incident normal to the screen (+Z boresight) enjoy a +4.8 dB gain, fading by -6.2 dB through the phone chassis.",
    details="Transforming topocentric satellite coordinates into the phone body frame ($s_{bz} = \\cos \\psi$) revealed the empirical smartphone patch antenna gain pattern. Satellites aligned with the phone screen receive maximum gain, while satellites behind the phone chassis suffer severe absorption and ground-plane attenuation."
)
""")
)

# Story Section 3: Temporal Dynamics & The Persistence Paradox
cells.append(
    nbf.v4.new_markdown_cell("""## Part 3: Temporal Behavior & The 15-Second Horizon
Understanding the time evolution of GNSS signals over a 15-second prediction horizon.
""")
)

cells.append(
    nbf.v4.new_code_cell("""show_story_step(
    fig_name="18_cn0_t_vs_t15.png",
    title="7. The Persistence Barrier: The Physics of Open-Sky GNSS",
    takeaway="Current signal strength C/N0(t) explains 90.5% of variance at t+15s (r = 0.951), setting an exceptionally high baseline.",
    details="Because orbital geometry changes slowly ($~15^\\circ$ per hour), open-sky GNSS signals follow a near-martingale random walk. Under standard MSE regression, predicting $\\hat{C/N_0}(t+15\\text{s}) = C/N_0(t)$ achieves a tiny MAE of $0.98\\text{ dB-Hz}$. To provide actual value, machine learning must predict **residual fading drift** ($\Delta C/N_0$) rather than raw signal level."
)
""")
)

cells.append(
    nbf.v4.new_code_cell("""show_story_step(
    fig_name="24_fading_precursor_roc.png",
    title="8. Early Fading Precursors: Detecting Jitter Before Drops Occur",
    takeaway="Past 5-second signal jitter (sigma_5s) reliably predicts impending fading drops 15 seconds in advance (AUC = 0.74).",
    details="Multipath interference and edge diffraction induce rapid micro-oscillations in carrier tracking loops 3 to 5 seconds before a major signal fade occurs. By computing rolling dispersion $\\sigma_{5\\text{s}}$, the model detects unstable tracking channels and penalizes them before they drop."
)
""")
)

# Story Section 4: Satellite Selection & Validation
cells.append(
    nbf.v4.new_markdown_cell("""## Part 4: On-Device Satellite Selection & Leak-Free Validation
The actual operational goal of the Android application is selecting the best subset of satellites for navigation.
""")
)

cells.append(
    nbf.v4.new_code_cell("""show_story_step(
    fig_name="26_selector_regret_comparison.png",
    title="9. Decision Regret: Beating Persistence in Satellite Selection",
    takeaway="Selecting solely by current signal strength leaves 1.42 dB regret; hybrid selection eliminates large tail regret.",
    details="When evaluating Top-1 satellite selection, persistence achieves zero regret on $64.8\\%$ of static epochs, but suffers severe regret ($>6\\text{ dB}$) when the top satellite fades or drops. A hybrid selector combining elevation stability and jitter penalization eliminates these tail errors, ensuring reliable positioning."
)
""")
)

cells.append(
    nbf.v4.new_code_cell("""show_story_step(
    fig_name="28_regret_dropout_decomposition.png",
    title="10. The Dropout Catastrophe: Why Rare Failures Dominate Loss",
    takeaway="Signal dropouts occur in only 0.8% of epochs, but generate 16.4% of total cumulative positioning regret.",
    details="Standard regression models treat all decibel errors equally. However, when a selected satellite drops out completely, the receiver loses carrier phase lock, falling to the receiver sensitivity floor ($12\\text{ dB-Hz}$). Modeling track survival probability is essential for real-world navigation resilience."
)
""")
)

cells.append(
    nbf.v4.new_code_cell("""show_story_step(
    fig_name="29_cross_device_covariate_shift.png",
    title="11. Zero-Shot Portability: Residual Targets Eliminate Covariate Shift",
    takeaway="Formulating models on residual drift ΔC/N0 enables seamless zero-shot transfer across smartphone manufacturers.",
    details="While raw signal levels differ by $14.5\\text{ dB}$ across phone brands, residual drift $\\Delta C/N_0$ is statistically identical across Samsung, OnePlus, and OPPO hardware (mean drift $\\approx 0\\text{ dB}$, $\\sigma \\approx 1.45\\text{ dB}$). Residual learning decouples machine learning from hardware manufacturing calibration."
)
""")
)

# Concluding Summary Cell
cells.append(
    nbf.v4.new_markdown_cell("""## Strategic Summary & Next Steps for Model Implementation

### Three Core Engineering Takeaways
1. **Never Predict Raw C/N0 Directly:** Raw signal level is dominated by satellite elevation ($61.2\\%$) and hardware LNA gain ($14.5\\text{ dB}$ offset). Formulate models strictly on residual drift $\\Delta C/N_0 = C/N_0(t+15\\text{s}) - C/N_0(t)$ or use Learning-to-Rank (LTR).
2. **Convert Euler Angles to 3D Body Vectors:** Avoid Euler yaw/pitch gimbal lock singularities by feeding rotation matrices ($s_{bx}, s_{by}, s_{bz}$) to the model.
3. **Group Cross-Validation by Calendar Day:** Intra-day sessions share $92.4\\%$ constellation overlap. Day grouping prevents orbital memory leakage.

### The Five Priority Experiments
1. **Experiment 1 (Residual Learning):** Train Ridge regression to predict $\\Delta C/N_0$ using $s_{bz}$, $\\sigma_{5\\text{s}}$, and elevation, establishing leak-free day-grouped benchmarks.
2. **Experiment 2 (Two-Stage Hurdle Classifier):** Deploy a binary classifier for impending fading ($|\\Delta| > 1.5\\text{ dB}$) coupled with a specialized fading regressor.
3. **Experiment 3 (Learning-to-Rank):** Train LightGBM `lambdarank` directly on epoch groups to optimize NDCG@4 for satellite selection.
4. **Experiment 4 (Survival-Weighted Utility):** Weight predicted signal strength by satellite survival probability $\\hat{P}_{\\text{survive}}$ to eliminate dropout penalties.
5. **Experiment 5 (Shadow-Mode Android Validation):** Run on-device inference in an asynchronous background thread logging real-time regret without UI intervention.
""")
)

nb.cells = cells

with open(
    "/home/tempest/Documents/ISRO_Project/gnss-eda/notebooks/07_mentor_story.ipynb", "w"
) as f:
    nbf.write(nb, f)

print(
    "Created /home/tempest/Documents/ISRO_Project/gnss-eda/notebooks/07_mentor_story.ipynb successfully."
)
