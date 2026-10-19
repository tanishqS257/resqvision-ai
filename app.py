import pandas as pd
import plotly.express as px
import os
import streamlit as st
import torch
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from streamlit_image_comparison import image_comparison
import matplotlib.pyplot as plt

# Import your model builders
from src.models.localization import build_localization_model
from src.models.damage_classifier import build_damage_model

# Page config
st.set_page_config(page_title="ResQVision AI", layout="wide")
st.title("ResQVision AI: Disaster Response Dashboard")
st.markdown("Automated building localization and damage severity assessment using satellite imagery.")

@st.cache_resource
def load_models():
    """Loads both AI models into GPU memory once."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Stage 1: Localization
    loc_model = build_localization_model(device=device)
    loc_model.load_state_dict(torch.load("artifacts/localization_best.pth", map_location=device))
    loc_model.eval()
    
    # Load Stage 2: Damage Classifier
    dam_model = build_damage_model(device=device)
    dam_model.load_state_dict(torch.load("artifacts/damage_classifier_best.pth", map_location=device))
    dam_model.eval()
    
    return loc_model, dam_model, device

loc_model, dam_model, device = load_models()

# Sidebar for image selection
st.sidebar.header("Select Disaster Event")
images_dir = "./data/train/images"
if os.path.exists(images_dir):
    all_files = os.listdir(images_dir)
    pre_images = sorted([f for f in all_files if f.endswith("_pre_disaster.png")])
    selected_pre = st.sidebar.selectbox("Choose a Pre-Disaster Image:", pre_images)
else:
    st.sidebar.error(f"Could not find dataset directory at {images_dir}")
    st.stop()

if selected_pre:
    # Resolve file paths
    selected_post = selected_pre.replace("_pre_disaster.png", "_post_disaster.png")
    pre_path = os.path.join(images_dir, selected_pre)
    post_path = os.path.join(images_dir, selected_post)
    
    # Load and resize images for the model
    pre_img = Image.open(pre_path).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    post_img = Image.open(post_path).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    
    # Layout: Top row for Image Slider
    st.subheader("1. Satellite Imagery Comparison")
    image_comparison(
        img1=pre_img,
        img2=post_img,
        label1="Pre-Disaster",
        label2="Post-Disaster",
        width=700,
        starting_position=50,
        show_labels=True,
        make_responsive=True
    )
    
    if st.button("Run AI Damage Assessment", type="primary"):
        with st.spinner("AI is analyzing structures and assessing damage..."):
            # Prepare Tensors
            pre_tensor = TF.to_tensor(pre_img).unsqueeze(0).to(device)
            post_tensor = TF.to_tensor(post_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Stage 1: Find buildings
                loc_logits = loc_model(pre_tensor)
                loc_mask = (torch.sigmoid(loc_logits).squeeze() > 0.5).cpu().numpy()
                
                # Stage 2: Assess Damage
                dam_logits = dam_model(pre_tensor, post_tensor)
                dam_preds = torch.argmax(dam_logits, dim=1).squeeze().cpu().numpy()
            
            # Combine logic: Apply damage predictions ONLY where buildings exist
            final_assessment = np.where(loc_mask == 1, dam_preds, 0)
            
            # Visualization Map
            color_map = {
                1: [0, 255, 0, 150],    # Green: No Damage
                2: [255, 255, 0, 150],  # Yellow: Minor Damage
                3: [255, 165, 0, 180],  # Orange: Major Damage
                4: [255, 0, 0, 200]     # Red: Destroyed
            }
            
            overlay = np.zeros((512, 512, 4), dtype=np.uint8)
            for damage_class, color in color_map.items():
                overlay[final_assessment == damage_class] = color
                
            overlay_img = Image.fromarray(overlay, "RGBA")
            result_display = post_img.copy().convert("RGBA")
            result_display.alpha_composite(overlay_img)
            
            # Display Results
            st.divider()
            st.subheader("2. AI Damage Assessment Results")
            col1, col2 = st.columns([3, 1])
            
            with col1:
               st.image(result_display, caption="Detected Buildings color-coded by Damage Severity", use_container_width=True)
                
            with col2:
                st.markdown("### 📊 Tactical Assessment Stats")
                
                # Calculate counts
                count_none = np.sum(final_assessment == 1)
                count_minor = np.sum(final_assessment == 2)
                count_major = np.sum(final_assessment == 3)
                count_destroyed = np.sum(final_assessment == 4)
                total_buildings = np.sum(loc_mask == 1)
                
                st.metric("Total Structures Identified", total_buildings)
                
                # Create a DataFrame for Plotly
                df_stats = pd.DataFrame({
                    "Severity": ["No Damage", "Minor", "Major", "Destroyed"],
                    "Count": [count_none, count_minor, count_major, count_destroyed]
                })
                
                # Generate a visually matching Pie Chart
                fig = px.pie(
                    df_stats, 
                    values="Count", 
                    names="Severity",
                    color="Severity",
                    color_discrete_map={
                        "No Damage": "#00FF00", 
                        "Minor": "#FFFF00", 
                        "Major": "#FFA500", 
                        "Destroyed": "#FF0000"
                    },
                    hole=0.4 # Makes it a donut chart
                )
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
                st.plotly_chart(fig, use_container_width=True)
                # 3. Export Capabilities
            st.divider()
            st.subheader("3. Intelligence Export")
            
            # Create a CSV of the summary
            csv_data = df_stats.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download BDA Summary Report (CSV)",
                data=csv_data,
                file_name=f"bda_report_{selected_pre.replace('.png', '')}.csv",
                mime="text/csv",
                type="primary"
            )