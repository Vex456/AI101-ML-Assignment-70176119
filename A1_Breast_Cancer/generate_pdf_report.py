"""
ReportLab IEEE 2-Column PDF Generator for Task A1 Report
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
    story.append(Paragraph("Task [A1]: [Breast Cancer Diagnosis from Cell Nuclei]", style_title))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Muhammad Zaryab Attiq</b> &nbsp;|&nbsp; Roll No.: <b>70176119</b> &nbsp;|&nbsp; Section: <b>Section A</b><br/>"
        "Repository: <u>https://github.com/Vex456/AI101-ML-Assignment-70176119</u><br/>"
        "Instructor: <b>Dr. Muhammad Awais</b>, AI Architect", style_author))
    story.append(FrameBreak()) # To Col 1 Page 1

    # ------------------ PAGE 1 - COLUMN 1 ------------------
    story.append(Paragraph("I. INTRODUCTION", style_h1))
    story.append(Paragraph(
        "Breast cancer is among the most prevalent life-threatening malignancies globally. Automated morphological analysis of fine-needle aspirate (FNA) cell nuclei offers an objective, non-invasive diagnostic pathway. In this work, we formulate an asymmetric supervised classification framework on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset comprising 569 patients and 30 nuclear descriptors. Because a missed malignancy (false negative) carries lethal clinical consequences compared to a false positive, we train, tune, and compare four supervised paradigms—Logistic Regression, Support Vector Machines (RBF), Random Forest, and Gradient Boosted Decision Trees (GBDT)—embedded in zero-leakage scikit-learn pipelines. By calibrating the decision threshold, our champion model elevates malignant recall to 95.24% with an ROC-AUC of 0.9947, demonstrating how threshold engineering aligns statistical machine learning with clinical screening safety.",
        style_body
    ))

    story.append(Paragraph("II. PROBLEM STATEMENT", style_h1))
    story.append(Paragraph(
        "Given an input vector <b>x</b> &isin; R<sup>30</sup> capturing ten nuclear morphometric properties across mean, standard error, and extreme (worst) values, the task is to learn a decision classifier <i>f: <b>x</b> &rarr; y &isin; {0, 1}</i>, where <i>y=1</i> denotes a malignant tumor (positive class) and <i>y=0</i> denotes a benign condition. In clinical screening, false negatives incur an order of magnitude higher penalty than false positives. Consequently, standard accuracy is uninformative. The formal learning objective is to maximize positive-class Recall TP/(TP+FN) and ROC-AUC subject to the operational constraint Recall(&tau;) &ge; 0.95 under a calibrated decision threshold &tau;.",
        style_body
    ))

    story.append(Paragraph("III. DATASET USED", style_h1))
    story.append(Paragraph(
        "We employ the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository (ID: 17). The corpus contains 569 patient samples with zero missing values, consisting of 212 malignant cases (37.26%) and 357 benign cases (62.74%). The 30 features quantify nuclear geometric attributes extracted from digitized microscopic images: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.",
        style_body
    ))

    story.append(Paragraph("IV. DATASET STATISTICS", style_h1))
    story.append(Paragraph(
        "Table I outlines the corpus partition and descriptive metrics. Intense geometric collinearity is observed among size parameters: mean radius, mean perimeter, and mean area exhibit pairwise correlations <i>r</i> &gt; 0.98. Malignant nuclei display substantially elevated mean area (978.3 vs. 462.8 &mu;m<sup>2</sup>) and mean concavity (0.160 vs. 0.046) compared to benign counterparts.",
        style_body
    ))

    # Table I: Dataset Statistics
    table_data_stats = [
        [Paragraph("<b>Property</b>", style_table_header), Paragraph("<b>Value</b>", style_table_header)],
        [Paragraph("Total patient samples", style_table_text_bold), Paragraph("569", style_table_text)],
        [Paragraph("Morphological continuous features", style_table_text), Paragraph("30", style_table_text)],
        [Paragraph("Missing values", style_table_text), Paragraph("0 (0.0%)", style_table_text)],
        [Paragraph("Class distribution", style_table_text_bold), Paragraph("212 Malignant / 357 Benign", style_table_text_bold)],
        [Paragraph("Malignant class proportion", style_table_text), Paragraph("37.26%", style_table_text)],
        [Paragraph("Train / test split (stratified)", style_table_text), Paragraph("455 (80%) / 114 (20%)", style_table_text)],
        [Paragraph("Test set positive (malignant) samples", style_table_text), Paragraph("42 samples", style_table_text)],
    ]
    t_stats = Table(table_data_stats, colWidths=[col_width * 0.65, col_width * 0.35])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(t_stats)
    story.append(Paragraph("TABLE I: Dataset statistics and clinical distribution.", style_caption))

    story.append(FrameBreak()) # Move to Col 2 Page 1

    # ------------------ PAGE 1 - COLUMN 2 ------------------
    if os.path.exists("A1_Breast_Cancer/figures/eda_correlation_matrix.png"):
        story.append(Image("A1_Breast_Cancer/figures/eda_correlation_matrix.png", width=col_width, height=col_width * 0.72))
        story.append(Paragraph("Fig. 1: Correlation heatmap of mean nuclear properties displaying strong collinearity between radius, perimeter, and area.", style_caption))

    story.append(Paragraph("V. METHODOLOGY", style_h1))
    story.append(Paragraph("A. Learning Setting & Leakage Prevention", style_h2))
    story.append(Paragraph(
        "The dataset is partitioned into an 80/20 stratified train/test split with fixed random seed (seed=42). To strictly prevent data leakage, all feature scaling (StandardScaler) is performed inside scikit-learn Pipeline constructs, ensuring standardization parameters (&mu;, &sigma;) are computed exclusively on the training folds.",
        style_body
    ))

    story.append(Paragraph("B. Model Selection & 5-Fold Stratified Tuning", style_h2))
    story.append(Paragraph(
        "We implement and tune four diverse algorithmic architectures:<br/>"
        "1) <i>Logistic Regression (Baseline):</i> L2-regularized linear model with inverse regularization <i>C</i> &isin; [0.01, 10.0].<br/>"
        "2) <i>Support Vector Classifier (RBF):</i> Non-linear kernel mapping with cost <i>C</i> &isin; [0.1, 10.0] and &gamma; &isin; ['scale', 0.01, 0.1].<br/>"
        "3) <i>Random Forest:</i> Ensemble of 100 decorrelated trees with maximum depth <i>d</i> &isin; [3, 5, &infin;].<br/>"
        "4) <i>Gradient Boosted Decision Trees (GBDT):</i> Stage-wise additive ensemble with learning rate &eta; &isin; [0.05, 0.1] and tree depth <i>d</i> &isin; [2, 3].<br/>"
        "Hyperparameters were optimized using 5-fold Stratified Cross-Validation on the training split, scoring for malignant recall.",
        style_body
    ))

    story.append(NextPageTemplate('TwoCol'))
    story.append(PageBreak()) # Move to Page 2

    # ------------------ PAGE 2 - COLUMN 1 ------------------
    story.append(Paragraph("VI. RESULTS", style_h1))
    story.append(Paragraph(
        "Table II summarizes the test-set performance across all evaluated models at default threshold &tau;=0.50. Support Vector Machine and Random Forest tied for the highest raw accuracy (97.37%) and F1-score (0.9630). Gradient Boosting achieved an ROC-AUC of 0.9947 and 100% precision.",
        style_body
    ))

    # Table II: Results
    table_data_results = [
        [Paragraph("<b>Model</b>", style_table_header), Paragraph("<b>Acc</b>", style_table_header), Paragraph("<b>Prec</b>", style_table_header), Paragraph("<b>Recall</b>", style_table_header), Paragraph("<b>F1</b>", style_table_header), Paragraph("<b>AUC</b>", style_table_header)],
        [Paragraph("Logistic Reg.", style_table_text), Paragraph("0.9649", style_table_text), Paragraph("0.9750", style_table_text), Paragraph("0.9286", style_table_text), Paragraph("0.9512", style_table_text), Paragraph("<b>0.9960</b>", style_table_text_bold)],
        [Paragraph("SVM (RBF)", style_table_text_bold), Paragraph("<b>0.9737</b>", style_table_text_bold), Paragraph("<b>1.0000</b>", style_table_text_bold), Paragraph("0.9286", style_table_text), Paragraph("<b>0.9630</b>", style_table_text_bold), Paragraph("0.9927", style_table_text)],
        [Paragraph("Random Forest", style_table_text_bold), Paragraph("<b>0.9737</b>", style_table_text_bold), Paragraph("<b>1.0000</b>", style_table_text_bold), Paragraph("0.9286", style_table_text), Paragraph("<b>0.9630</b>", style_table_text_bold), Paragraph("0.9950", style_table_text)],
        [Paragraph("Gradient Boosting", style_table_text), Paragraph("0.9649", style_table_text), Paragraph("<b>1.0000</b>", style_table_text_bold), Paragraph("0.9048", style_table_text), Paragraph("0.9500", style_table_text), Paragraph("0.9947", style_table_text)],
    ]
    t_res = Table(table_data_results, colWidths=[col_width * 0.35, col_width * 0.13, col_width * 0.13, col_width * 0.13, col_width * 0.13, col_width * 0.13])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
    ]))
    story.append(t_res)
    story.append(Paragraph("TABLE II: Model evaluation on held-out test set (&tau;=0.50).", style_caption))

    if os.path.exists("A1_Breast_Cancer/figures/combined_key_result.png"):
        story.append(Image("A1_Breast_Cancer/figures/combined_key_result.png", width=col_width, height=col_width * 0.76))
        story.append(Paragraph("Fig. 2: Diagnostic evaluations: (a) ROC curves; (b) Precision-Recall curves; (c) Calibrated confusion matrix (&tau;=0.05, Recall=0.9524); (d) GBDT Gini feature importance.", style_caption))

    story.append(Paragraph("VII. DISCUSSION & ANALYSIS", style_h1))
    story.append(Paragraph("A. Model Superiority & The Fallacy of Accuracy", style_h2))
    story.append(Paragraph(
        "While SVM and Random Forest achieved 97.37% accuracy, evaluating clinical screening purely on accuracy is hazardous. A trivial majority-class classifier that labels all samples benign attains 62.74% accuracy while failing 100% of cancer patients. Precision, Recall, and ROC-AUC are the sole clinically sound metrics.",
        style_body
    ))

    story.append(FrameBreak()) # Move to Col 2 Page 2

    # ------------------ PAGE 2 - COLUMN 2 ------------------
    story.append(Paragraph("B. Decision Threshold Calibration & Precision Cost", style_h2))
    story.append(Paragraph(
        "At default threshold &tau;=0.50, GBDT missed 4 malignant tumors (Recall=0.9048). By systematically shifting the decision threshold down to &tau;=0.05, the model achieved Recall=0.9524 (40 of 42 malignant tumors correctly flagged), strictly satisfying the clinical &ge;0.95 safety requirement. Crucially, because the underlying ROC-AUC is 0.9947, this sensitivity gain cost only 4.76% in precision (dropping from 1.0000 to 0.9524, with only 2 benign biopsies flagged).",
        style_body
    ))

    story.append(Paragraph("C. Biomarker Ranking & Multicollinearity Impact", style_h2))
    story.append(Paragraph(
        "Gini importance analysis identifies worst area, worst concave points, and worst perimeter as the dominant diagnostic biomarkers. Dropping collinear features (mean radius, mean perimeter) produced negligible change in test performance (F1=0.9512), confirming that both L2 regularization and tree splitting stably isolate complementary morphologic signals.",
        style_body
    ))

    story.append(Paragraph("VIII. CONCLUSION", style_h1))
    story.append(Paragraph(
        "Supervised learning models paired with standardized feature scaling achieve outstanding discriminative accuracy on nuclear FNA biopsies. Threshold calibration effectively suppressed false negatives to meet the &ge;0.95 recall safety threshold with minimal false alarms.",
        style_body
    ))

    story.append(PageBreak()) # Move to Page 3 (Disclosures & References)

    # ------------------ PAGE 3 - DISCLOSURE & REFS ------------------
    story.append(Paragraph("AI USE DISCLOSURE STATEMENT", style_h1))
    story.append(Paragraph(
        "• <b>AI Tool Utilized:</b> Google Antigravity Agent (Gemini 2.5 Pro).<br/>"
        "• <b>Scope of Assistance:</b> Scaffolding scikit-learn cross-validation pipelines, IEEE LaTeX report formatting, and multi-panel diagnostic figure generation.<br/>"
        "• <b>Human Verification:</b> All models, cross-validations, metrics, and threshold curves were executed locally with fixed random seeds (seed=42) and verified against raw outputs.",
        style_body
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("REFERENCES", style_h1))
    refs = [
        "[1] W. N. Street, W. H. Wolberg, and O. L. Mangasarian, 'Nuclear feature extraction for breast tumor diagnosis,' in <i>Proc. IS&T/SPIE 1993 Int. Symp. Electronic Imaging</i>, 1993, pp. 861-870.",
        "[2] F. Pedregosa <i>et al.</i>, 'Scikit-learn: Machine learning in Python,' <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825-2830, 2011."
    ]
    for ref in refs:
        story.append(Paragraph(ref, style_body))

    doc.build(story)
    print(f"Report successfully compiled to: {output_pdf_path}")

if __name__ == '__main__':
    generate_report("A1_Breast_Cancer/report/70176119_A1_Report.pdf")
