"""
ReportLab IEEE 2-Column PDF Generator for Task A3 Report
Generates an A4, double-column, IEEE conference style PDF report strictly adhering to the 2-page content limit.
Student Name: Muhammad Zaryab Attiq | Roll No: 70176119 | Section: Section A
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
    Table, TableStyle, Image, FrameBreak, NextPageTemplate, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

def generate_report(output_pdf_path):
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    page_width, page_height = A4
    margin_x = 36
    margin_top = 40
    margin_bottom = 40
    gutter = 14
    content_width = page_width - 2 * margin_x
    col_width = (content_width - gutter) / 2
    
    # Page 1 Frames: Header + 2 columns
    header_height = 80
    frame_header = Frame(margin_x, page_height - margin_top - header_height, content_width, header_height, id='col_header', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)
    col1_p1 = Frame(margin_x, margin_bottom, col_width, page_height - margin_top - header_height - margin_bottom, id='col1_p1', topPadding=4, bottomPadding=0, leftPadding=0, rightPadding=0)
    col2_p1 = Frame(margin_x + col_width + gutter, margin_bottom, col_width, page_height - margin_top - header_height - margin_bottom, id='col2_p1', topPadding=4, bottomPadding=0, leftPadding=0, rightPadding=0)

    # Full Two-Column Frames
    col_height_full = page_height - margin_top - margin_bottom
    col1_full = Frame(margin_x, margin_bottom, col_width, col_height_full, id='col1_full', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)
    col2_full = Frame(margin_x + col_width + gutter, margin_bottom, col_width, col_height_full, id='col2_full', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)

    doc = BaseDocTemplate(output_pdf_path, pagesize=A4, leftMargin=margin_x, rightMargin=margin_x, topMargin=margin_top, bottomMargin=margin_bottom)
    
    template_page1 = PageTemplate(id='Page1', frames=[frame_header, col1_p1, col2_p1])
    template_two_col = PageTemplate(id='TwoCol', frames=[col1_full, col2_full])
    doc.addPageTemplates([template_page1, template_two_col])

    styles = getSampleStyleSheet()
    style_supertitle = ParagraphStyle('SuperTitle', fontName='Helvetica-Bold', fontSize=7.5, leading=9, alignment=TA_CENTER, textColor=colors.HexColor('#334155'))
    style_title = ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=14, leading=17, alignment=TA_CENTER, textColor=colors.HexColor('#0f172a'))
    style_author = ParagraphStyle('Author', fontName='Helvetica', fontSize=8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor('#1e293b'))
    style_h1 = ParagraphStyle('SectionHeading', fontName='Helvetica-Bold', fontSize=9.5, leading=12, spaceBefore=6, spaceAfter=2, textColor=colors.HexColor('#0f172a'))
    style_h2 = ParagraphStyle('SubSectionHeading', fontName='Helvetica-Bold', fontSize=8.5, leading=11, spaceBefore=4, spaceAfter=1, textColor=colors.HexColor('#1e293b'))
    style_body = ParagraphStyle('BodyTextJustified', fontName='Times-Roman', fontSize=8.2, leading=10.2, alignment=TA_JUSTIFY, spaceAfter=4, textColor=colors.HexColor('#0f172a'))
    style_table_text = ParagraphStyle('TableText', fontName='Times-Roman', fontSize=7.0, leading=8.2, alignment=TA_CENTER)
    style_table_text_bold = ParagraphStyle('TableTextBold', fontName='Helvetica-Bold', fontSize=7.0, leading=8.2, alignment=TA_CENTER)
    style_table_header = ParagraphStyle('TableHeader', fontName='Helvetica-Bold', fontSize=7.2, leading=8.5, alignment=TA_CENTER, textColor=colors.white)
    style_caption = ParagraphStyle('Caption', fontName='Helvetica-Oblique', fontSize=7.0, leading=8.2, alignment=TA_CENTER, textColor=colors.HexColor('#334155'), spaceBefore=2, spaceAfter=4)

    story = []

    # ------------------ FIRST PAGE HEADER ------------------
    story.append(Paragraph("THE UNIVERSITY OF LAHORE • DEPARTMENT OF INTELLIGENT SYSTEMS • AI-101", style_supertitle))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Task [A3]: [Dry Bean Variety Recognition from Shape Features]", style_title))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Muhammad Zaryab Attiq</b> &nbsp;|&nbsp; Roll No.: <b>70176119</b> &nbsp;|&nbsp; Section: <b>Section A</b><br/>"
        "Repository: <u>https://github.com/Vex456/AI101-ML-Assignment-70176119</u><br/>"
        "Instructor: <b>Dr. Muhammad Awais</b>, AI Architect", style_author))
    story.append(FrameBreak()) # To Col 1 Page 1

    # ------------------ PAGE 1 - COLUMN 1 ------------------
    story.append(Paragraph("I. INTRODUCTION", style_h1))
    story.append(Paragraph(
        "Automated optical seed classification plays a crucial role in modern agrarian supply chains to certify variety purity and market grading. In this study, we formulate a multi-class supervised learning benchmark on 13,611 dry bean specimens across seven registered cultivars (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) using 16 morphometric features. We construct zero-leakage scikit-learn classification pipelines, evaluating multinomial logistic regression, class-weighting strategies, random forests, gradient boosting, and multi-layer perceptrons (MLP). Our champion models achieve a Macro-F1 score of 0.9357 (92.31% test accuracy), while principal component analysis (PCA) reveals that only 8 orthogonal dimensions are required to retain 99.7% of full-space classification capacity.",
        style_body
    ))

    story.append(Paragraph("II. PROBLEM STATEMENT", style_h1))
    story.append(Paragraph(
        "Given an input vector <b>x</b> &isin; R<sup>16</sup> describing 12 morphological dimensions (area, perimeter, axis lengths, compactness) and 4 structural shape factors, the objective is to predict the discrete bean variety <i>y &isin; {1, 2, ..., 7}</i>. Because the class distribution exhibits moderate natural skew (ranging from 3,546 Dermason to 522 Bombay seeds), overall accuracy alone is inadequate. Success is defined primarily by the Macro-Averaged F1-Score: Macro-F1 = (1/7) &Sigma; F<sub>1,c</sub> alongside Weighted-F1, per-class precision and recall, and a 7&times;7 confusion matrix.",
        style_body
    ))

    story.append(Paragraph("III. DATASET USED", style_h1))
    story.append(Paragraph(
        "We utilize the Dry Bean Dataset from the UCI Machine Learning Repository (ID: 602). The dataset comprises 13,611 high-resolution grain images segmented into 16 continuous shape features without missing entries. The seven target classes are: Dermason (26.05%), Sira (19.37%), Seker (14.89%), Horoz (14.16%), Cali (11.98%), Barbunya (9.71%), and Bombay (3.84%).",
        style_body
    ))

    story.append(Paragraph("IV. DATASET STATISTICS", style_h1))
    story.append(Paragraph(
        "Table I summarizes dataset and partition statistics. High collinearity is prevalent across geometric scale features: Area and ConvexArea correlate at <i>r</i>=1.00, while Perimeter and MajorAxisLength exhibit <i>r</i> &gt; 0.97. A 2D PCA projection illustrates clean separation for Bombay (large flat seeds) and Horoz (elongated seeds), with notable boundary overlap between Dermason and Sira.",
        style_body
    ))

    # Table I: Dataset Statistics
    table_data_stats = [
        [Paragraph("<b>Property</b>", style_table_header), Paragraph("<b>Value</b>", style_table_header)],
        [Paragraph("Total grain samples", style_table_text_bold), Paragraph("13,611", style_table_text)],
        [Paragraph("Continuous shape features", style_table_text), Paragraph("16", style_table_text)],
        [Paragraph("Number of target classes", style_table_text), Paragraph("7 registered cultivars", style_table_text)],
        [Paragraph("Class distribution range", style_table_text_bold), Paragraph("Dermason (3,546) to Bombay (522)", style_table_text_bold)],
        [Paragraph("Data split ratio (stratified)", style_table_text), Paragraph("70% / 15% / 15%", style_table_text)],
        [Paragraph("Train / validation / test counts", style_table_text), Paragraph("9,527 / 2,042 / 2,042", style_table_text)],
        [Paragraph("Feature preprocessing", style_table_text), Paragraph("StandardScaler (train-fitted)", style_table_text)],
    ]
    t_stats = Table(table_data_stats, colWidths=[col_width * 0.62, col_width * 0.38])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(t_stats)
    story.append(Paragraph("TABLE I: Dataset statistics and split breakdown.", style_caption))

    story.append(FrameBreak()) # Move to Col 2 Page 1

    # ------------------ PAGE 1 - COLUMN 2 ------------------
    if os.path.exists("A3_Dry_Bean/figures/eda_pca_scatter.png"):
        story.append(Image("A3_Dry_Bean/figures/eda_pca_scatter.png", width=col_width, height=col_width * 0.68))
        story.append(Paragraph("Fig. 1: 2D PCA projection of the 16 shape descriptors colored by bean cultivar (capturing 82.0% cumulative variance).", style_caption))

    story.append(Paragraph("V. METHODOLOGY", style_h1))
    story.append(Paragraph("A. Learning Setting & Leakage Prevention", style_h2))
    story.append(Paragraph(
        "The corpus is split into a stratified 70/15/15 train/validation/test partition (seed=42). All feature standardization (StandardScaler) is performed inside a scikit-learn Pipeline fitted strictly on the 70% training fold (zero validation/test leakage).",
        style_body
    ))

    story.append(Paragraph("B. Model Selection & Class Weighting Study", style_h2))
    story.append(Paragraph(
        "We evaluate five distinct architectural configurations:<br/>"
        "1) <i>Multinomial Logistic Regression:</i> Unweighted baseline with L2 penalty.<br/>"
        "2) <i>Class-Weighted Logistic Regression:</i> Inverse-frequency weighting (class_weight='balanced').<br/>"
        "3) <i>Random Forest:</i> Ensemble of 100 decorrelated trees with max depth 12.<br/>"
        "4) <i>Multi-Layer Perceptron (MLP):</i> Feed-forward neural network with two hidden layers (100, 50) and ReLU activations.<br/>"
        "5) <i>Gradient Boosted Decision Trees (GBDT):</i> 100 boosting stages with learning rate 0.1 and max depth 3.<br/>"
        "To assess feature redundancy, we project scaled inputs onto the first <i>k</i> &isin; {2, 4, 8, 16} principal components and track Macro-F1 retention.",
        style_body
    ))

    story.append(NextPageTemplate('TwoCol'))
    story.append(PageBreak()) # Move to Page 2

    # ------------------ PAGE 2 - COLUMN 1 ------------------
    story.append(Paragraph("VI. RESULTS", style_h1))
    story.append(Paragraph(
        "Table II summarizes final model evaluations on the unseen test set. The Multi-Layer Perceptron and Random Forest achieved top classification metrics, securing 0.9357 and 0.9348 Macro-F1 (92.31% accuracy), respectively.",
        style_body
    ))

    # Table II: Results
    table_data_results = [
        [Paragraph("<b>Model Architecture</b>", style_table_header), Paragraph("<b>Accuracy</b>", style_table_header), Paragraph("<b>Macro-F1</b>", style_table_header), Paragraph("<b>Weighted-F1</b>", style_table_header)],
        [Paragraph("Multinomial Logistic Reg.", style_table_text), Paragraph("0.9207", style_table_text), Paragraph("0.9338", style_table_text), Paragraph("0.9207", style_table_text)],
        [Paragraph("Logistic Reg. (Balanced)", style_table_text), Paragraph("0.9187", style_table_text), Paragraph("0.9331", style_table_text), Paragraph("0.9188", style_table_text)],
        [Paragraph("Random Forest", style_table_text_bold), Paragraph("<b>0.9231</b>", style_table_text_bold), Paragraph("0.9348", style_table_text), Paragraph("<b>0.9231</b>", style_table_text_bold)],
        [Paragraph("Multi-Layer Perceptron (MLP)", style_table_text_bold), Paragraph("<b>0.9231</b>", style_table_text_bold), Paragraph("<b>0.9357</b>", style_table_text_bold), Paragraph("0.9228", style_table_text)],
        [Paragraph("Gradient Boosting", style_table_text), Paragraph("0.9202", style_table_text), Paragraph("0.9328", style_table_text), Paragraph("0.9201", style_table_text)],
    ]
    t_res = Table(table_data_results, colWidths=[col_width * 0.46, col_width * 0.18, col_width * 0.18, col_width * 0.18])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
    ]))
    story.append(t_res)
    story.append(Paragraph("TABLE II: Classification performance on held-out test set.", style_caption))

    if os.path.exists("A3_Dry_Bean/figures/combined_key_result.png"):
        story.append(Image("A3_Dry_Bean/figures/combined_key_result.png", width=col_width, height=col_width * 0.44))
        story.append(Paragraph("Fig. 2: (a) 7x7 confusion matrix on test set; (b) PCA dimensionality retention curve across k=2, 4, 8, 16.", style_caption))

    story.append(Paragraph("VII. DISCUSSION & ANALYSIS", style_h1))
    story.append(Paragraph("A. Variety Confusion Dynamics (Sira vs. Dermason)", style_h2))
    story.append(Paragraph(
        "The confusion matrix in Fig. 2(a) indicates that misclassifications concentrate heavily between Sira and Dermason (97 Dermason predicted as Sira, and 58 Sira predicted as Dermason). These cultivars possess overlapping morphologic profiles: both are small, compact white beans with identical median aspect ratios (1.51 vs. 1.50) and closely clustered perimeter distributions, establishing an optical resolution boundary.",
        style_body
    ))

    story.append(FrameBreak()) # Move to Col 2 Page 2

    # ------------------ PAGE 2 - COLUMN 2 ------------------
    story.append(Paragraph("B. Impact of Class Weighting", style_h2))
    story.append(Paragraph(
        "Comparing unweighted versus balanced logistic regression on the validation split shows that unweighted regression achieves 0.9339 accuracy and 0.9446 Macro-F1, whereas balanced weighting yields 0.9310 accuracy and 0.9438 Macro-F1. Reweighting caused a 0.29% accuracy decline because the minority Bombay class already has 522 samples—ample for geometric bounding—making uniform loss optimization globally optimal.",
        style_body
    ))

    story.append(Paragraph("C. PCA Dimensionality Retention Threshold", style_h2))
    story.append(Paragraph(
        "Tracking classification performance across <i>k</i> &isin; {2, 4, 8, 16} components reveals: <i>k</i>=2 yields F1=0.8658 (82.0% variance); <i>k</i>=4 yields F1=0.9009 (95.0% variance); and <i>k</i>=8 yields F1=0.9317 (99.9% variance). The 98% retention threshold corresponds to 0.9346 &times; 0.98 = 0.9159. Consequently, <i>k</i>=8 components comfortably exceed the 98% target, compressing feature dimensionality by 50% with virtually zero loss in predictive fidelity.",
        style_body
    ))

    story.append(Paragraph("VIII. CONCLUSION", style_h1))
    story.append(Paragraph(
        "Supervised neural and tree models accurately classify dry bean cultivars from 16 morphometric features. PCA demonstrates that half the feature space captures over 99.7% of discriminative performance, validating efficient automated seed sorting pipelines.",
        style_body
    ))

    # ------------------ AI DISCLOSURE & REFS (FITS ON PAGE 2) ------------------
    story.append(Paragraph("AI USE DISCLOSURE STATEMENT", style_h1))
    story.append(Paragraph(
        "• <b>AI Tool Utilized:</b> Google Antigravity Agent (Gemini 2.5 Pro).<br/>"
        "• <b>Scope of Assistance:</b> Scaffolding scikit-learn cross-validation pipelines, PCA dimensionality analysis, and IEEE two-column report compilation.<br/>"
        "• <b>Human Verification:</b> All models and PCA evaluations were run locally with deterministic random seeds (seed=42); all reported metrics match notebook outputs 1:1.",
        style_body
    ))

    story.append(Paragraph("REFERENCES", style_h1))
    refs = [
        "[1] M. Koklu and I. A. Ozkan, 'Multiclass classification of dry beans using computer vision and machine learning techniques,' <i>Comput. Electron. Agric.</i>, vol. 174, p. 105507, 2020.",
        "[2] F. Pedregosa <i>et al.</i>, 'Scikit-learn: Machine learning in Python,' <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825-2830, 2011."
    ]
    for ref in refs:
        story.append(Paragraph(ref, style_body))

    doc.build(story)
    print(f"Report successfully compiled to: {output_pdf_path}")

if __name__ == '__main__':
    generate_report("A3_Dry_Bean/report/70176119_A3_Report.pdf")
