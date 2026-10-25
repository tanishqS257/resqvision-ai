from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors

def generate_pdf_report(filename="ResQVision_AI_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=15,
        spaceBefore=15,
        textColor=colors.maroon
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=10,
        leading=16,
        alignment=TA_JUSTIFY
    )
    
    story = []

    # PAGE 1: Title and Executive Summary
    story.append(Paragraph("ResQVision AI: Technical Architecture", title_style))
    story.append(Paragraph("Battle Damage Assessment & Disaster Response", styles['Heading3']))
    story.append(Spacer(1, 40))
    story.append(Paragraph("1. Executive Summary", heading_style))
    story.append(Paragraph(
        "ResQVision AI is a specialized, end-to-end Machine Learning pipeline engineered for "
        "automated structural damage assessment using pre- and post-disaster satellite imagery. "
        "Designed primarily for high-stakes operational environments, the system translates seamlessly into "
        "military Battle Damage Assessment (BDA) by identifying rapid structural destruction caused "
        "by heavy weaponry or natural events.", body_style
    ))
    story.append(Paragraph(
        "By automating high-resolution imagery inspection, the system drastically reduces cognitive "
        "load on tactical analysts. This report details the two-stage computer vision "
        "architecture, the dataset preparation, the hyperparameter configurations, and the "
        "interactive intelligence dashboard deployed for operational decision-making.", body_style
    ))
    story.append(PageBreak())

    # PAGE 2: Two-Stage Model Architecture
    story.append(Paragraph("2. Model Architecture: Two-Stage Siamese Network", heading_style))
    story.append(Paragraph(
        "Direct pixel-level damage classification across raw imagery is error-prone because "
        "deep neural networks frequently confuse background environmental shifts with targeted "
        "structural degradation. ResQVision AI resolves this using a modular Two-Stage Pipeline.", body_style
    ))
    story.append(Paragraph("Stage 1: Building Footprint Localization", styles['Heading3']))
    story.append(Paragraph(
        "The first stage isolates and registers building footprints using strictly the pre-event image. "
        "It deploys a U-Net architecture featuring an ImageNet-pretrained ResNet34 encoder. The network "
        "outputs a calibrated binary mask distinguishing background (0) from structure (1).", body_style
    ))
    story.append(Paragraph("Stage 2: Damage Severity Classification", styles['Heading3']))
    story.append(Paragraph(
        "The second stage evaluates structural degradation using a Twin-Tower Substrate (TTS) "
        "Siamese network. Pre- and post-event images are processed through shared encoder weights. "
        "The architecture computes element-wise feature differences (|post - pre|) at each scale "
        "to expose localized structural alterations, producing a 5-class severity map: "
        "Background, No Damage, Minor Damage, Major Damage, and Destroyed.", body_style
    ))
    story.append(PageBreak())

    # PAGE 3: Dataset Configuration & Hyperparameters
    story.append(Paragraph("3. Dataset Configuration & Training Protocol", heading_style))
    story.append(Paragraph(
        "The project is trained on the xView2 (xBD) dataset. To respect RTX 2050 4GB VRAM limits, "
        "satellite tiles are resized via Bilinear interpolation to 512x512. The data partition maintains "
        "an 80% training split (800 image pairs) and a 20% validation split (200 image pairs).", body_style
    ))
    story.append(Paragraph("Key Hyperparameters:", styles['Heading3']))
    items = [
        ListItem(Paragraph("<b>Training Epochs:</b> 5 complete optimization cycles.", body_style)),
        ListItem(Paragraph("<b>Batch Size:</b> 4 paired image tiles per iteration.", body_style)),
        ListItem(Paragraph("<b>Optimizer:</b> AdamW with learning rate 1e-4.", body_style)),
        ListItem(Paragraph("<b>Loss Function:</b> Weighted CrossEntropyLoss.", body_style))
    ]
    story.append(ListFlowable(items, bulletType='bullet'))
    story.append(Paragraph(
        "To mitigate heavy class imbalance where undamaged structures dominate, class loss weights "
        "are scaled to heavily penalize errors on destroyed objects: Background (0.2), "
        "No Damage (1.0), Minor (2.0), Major (3.0), and Destroyed (4.0).", body_style
    ))
    story.append(PageBreak())

    # PAGE 4: Tactical Intelligence Dashboard
    story.append(Paragraph("4. Tactical Intelligence Dashboard", heading_style))
    story.append(Paragraph(
        "The dual model checkpoints (localization_best.pth and damage_classifier_best.pth) "
        "are integrated into a streamlined Streamlit application designed for rapid field deployment.", body_style
    ))
    story.append(Paragraph("Integrated Functional Modules:", styles['Heading3']))
    dash_items = [
        ListItem(Paragraph("<b>Dual-Pane Slider:</b> Interactive split-view slider for side-by-side pre- and post-event satellite inspection.", body_style)),
        ListItem(Paragraph("<b>Dynamic Analytics:</b> Plotly donut visualizations aggregating localized structural damage metrics.", body_style)),
        ListItem(Paragraph("<b>Threshold-Based Alerts:</b> Automatic classification of target zones into Critical, Elevated, or Stable threat levels.", body_style)),
        ListItem(Paragraph("<b>Data Export:</b> Standardized CSV export for downstream mission intelligence integration.", body_style))
    ]
    story.append(ListFlowable(dash_items, bulletType='bullet'))
    story.append(PageBreak())

    # PAGE 5: Quantitative Evaluation & Conclusion
    story.append(Paragraph("5. Quantitative Evaluation & Conclusion", heading_style))
    story.append(Paragraph(
        "To establish academic and operational validity, evaluation scripts generate standard "
        "computer vision benchmarks: Intersection over Union (IoU) for Stage 1 localization and "
        "Macro-averaged F1-Score across the discrete damage categories.", body_style
    ))
    story.append(Paragraph("Conclusion", styles['Heading3']))
    story.append(Paragraph(
        "ResQVision AI establishes an end-to-end framework translating complex remote sensing "
        "data into practical, field-grade intelligence. Combining a Twin-Tower Siamese backbone "
        "with an accessible analytical dashboard produces an effective, production-ready solution "
        "for Battle Damage Assessment and disaster mitigation workflows.", body_style
    ))

    doc.build(story)
    print(f"Success! {filename} generated successfully.")

if __name__ == "__main__":
    generate_pdf_report()
