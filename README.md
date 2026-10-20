# ResQVision AI 🛰️

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

**ResQVision AI** is an end-to-end ML model related to military and tactical intelligence dashboard designed for automated Battle Damage Assessment (BDA) and rapid disaster response[cite: 1]. By processing high-resolution satellite imagery, the system locates man-made structures and evaluates their structural integrity following heavy weaponry strikes or natural disasters.

## 🧠 Architecture: Two-Stage Siamese Network
Predicting damage directly from raw imagery is highly inaccurate due to environmental noise. This project mitigates that using a two-stage computer vision architecture:

1. **Building Localization (Stage 1):** A U-Net with a pre-trained ResNet34 encoder isolates and extracts building footprints strictly from the pre-event imagery.
2. **Damage Severity Classification (Stage 2):** A Twin-Tower Substrate (TTS) Siamese U-Net processes both the pre- and post-event images. It computes the element-wise absolute difference across feature maps to isolate structural changes, outputting a 5-class severity mask (Background, No Damage, Minor Damage, Major Damage, Destroyed).

## 📊 Dataset & Processing
This model was trained on the **xView2 (xBD) Challenge Dataset**. 
* **Input Resolution:** Resized to 512x512 to accommodate hardware limits.
* **Class Balancing:** Utilizes weighted CrossEntropyLoss to heavily penalize errors on destroyed structures, combatting the natural imbalance of undamaged buildings in satellite imagery.

## 🚀 Live Dashboard Features
The final model weights are deployed via Streamlit Community Cloud for tactical analysts. 
* **Dual-Pane Inspection:** Interactive before/after visual slider.
* **Dynamic Analytics:** Automated aggregation of structural damage metrics via Plotly.
* **Threat Level Alerting:** Algorithmically flags operational zones as *Critical*, *Elevated*, or *Stable*.
* **Intelligence Export:** One-click CSV extraction for downstream mission planning.

## 💻 Local Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/tanishqS257/resqvision-ai.git](https://github.com/tanishqS257/resqvision-ai.git)
   cd resqvision-ai
```bash
pip install -r requirements.txt

**For running the dashboard:**
```bash
streamlit run app.py

**For generating the PDF:**
```bash
python generate_bda_report.py
