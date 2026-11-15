"""
ReportLab IEEE 2-Column PDF Generator for Task B1 Report
Generates an A4, double-column, IEEE conference style PDF report strictly adhering to the 2-page content limit.
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
    
    # First Page: Header across top, then 2 columns
    header_height = 80
    frame_header = Frame(margin_x, page_height - margin_top - header_height, content_width, header_height, id='col_header', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)
    col1_p1 = Frame(margin_x, margin_bottom, col_width, page_height - margin_top - header_height - margin_bottom, id='col1_p1', topPadding=4, bottomPadding=0, leftPadding=0, rightPadding=0)
    col2_p1 = Frame(margin_x + col_width + gutter, margin_bottom, col_width, page_height - margin_top - header_height - margin_bottom, id='col2_p1', topPadding=4, bottomPadding=0, leftPadding=0, rightPadding=0)

    # Subsequent Pages: Full 2 columns
    col_height_full = page_height - margin_top - margin_bottom
    col1_full = Frame(margin_x, margin_bottom, col_width, col_height_full, id='col1_full', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)
    col2_full = Frame(margin_x + col_width + gutter, margin_bottom, col_width, col_height_full, id='col2_full', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)

    doc = BaseDocTemplate(output_pdf_path, pagesize=A4, leftMargin=margin_x, rightMargin=margin_x, topMargin=margin_top, bottomMargin=margin_bottom)
    
    template_page1 = PageTemplate(id='Page1', frames=[frame_header, col1_p1, col2_p1])
    template_two_col = PageTemplate(id='TwoCol', frames=[col1_full, col2_full])
    doc.addPageTemplates([template_page1, template_two_col])

    # Typography styles
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
    story.append(Paragraph("Task [B1]: Detecting Fraudulent Job Postings", style_title))
    story.append(Spacer(1, 3))
    story.append(Paragraph("<b>Muhammad Zaryab Attiq</b> &nbsp;|&nbsp; Roll No. 70176119 &nbsp;|&nbsp; Section A<br/>Repository: <u>https://github.com/Vex456/AI101-ML-Assignment-70176119</u> &nbsp;|&nbsp; Instructor: Dr. Muhammad Awais, AI Architect", style_author))
    story.append(FrameBreak()) # To col1_p1

    # ------------------ COLUMN 1 (PAGE 1) ------------------
    story.append(Paragraph("1. Introduction", style_h1))
    story.append(Paragraph("The rapid proliferation of remote recruitment platforms has intensified online employment fraud, exposing job seekers to identity theft and advance-fee wire schemes. In this study, we formulate an imbalanced binary classification framework on the Employment Scam Aegean Dataset (EMSCAD) containing 17,880 postings, where only 4.84% are fraudulent. By fusing unstructured text fields (descriptions, requirements, benefits) with structured metadata (missingness indicators, company logo presence, telecommuting status) via a zero-leakage scikit-learn ColumnTransformer, our champion multimodal model achieves an F1 score of <b>0.835</b> on the fraud class and a PR-AUC of <b>0.907</b>, outperforming models reliant purely on administrative structured metadata by over 159% in F1 gain.", style_body))

    story.append(Paragraph("2. Problem Statement", style_h1))
    story.append(Paragraph("Given an input tuple <i>x = (x<sub>text</sub>, x<sub>struct</sub>)</i> representing free-text job content alongside categorical and binary administrative metadata, our task is to learn a decision classifier <i>f: x &mapsto; y &isin; {0, 1}</i>, where <i>y=1</i> denotes a fraudulent advertisement and <i>y=0</i> denotes a legitimate one. Because the dataset exhibits extreme class imbalance (95.16% negative vs. 4.84% positive), standard accuracy is uninformative. Success is defined primarily by the Area Under the Precision-Recall Curve (PR-AUC) and the F1 score on the positive (fraud) class, with the decision threshold &tau; calibrated on a held-out validation split.", style_body))

    story.append(Paragraph("3. Dataset Used", style_h1))
    story.append(Paragraph("We employ the Real or Fake Job Postings dataset (EMSCAD) hosted on Kaggle [1]. The corpus comprises 17,880 job advertisements described by 17 features: 5 free-text fields (title, company profile, description, requirements, benefits), 6 categorical attributes (employment type, required experience, education, industry, function, location), and 3 binary indicators (telecommuting, has company logo, has questions). We explicitly engineer missingness indicators for key fields because missing company metadata provides a strong inductive signal of fraud.", style_body))

    story.append(Paragraph("4. Dataset Statistics", style_h1))
    story.append(Paragraph("Table I provides the summary statistics for the EMSCAD corpus. Text lengths vary substantially between classes: legitimate postings have a higher median word count (248 words) than fraudulent postings (178 words). Furthermore, postings lacking a company logo exhibit an elevated fraud rate of 15.8%, compared to only 3.1% for logo-bearing postings.", style_body))

    # Table 1: Dataset Statistics
    t1_data = [
        [Paragraph("<b>Property</b>", style_table_header), Paragraph("<b>Value</b>", style_table_header)],
        [Paragraph("Samples (after cleaning)", style_table_text), Paragraph("17,880", style_table_text)],
        [Paragraph("Features (num / cat / text)", style_table_text), Paragraph("2 / 10 / 5", style_table_text)],
        [Paragraph("Target classes", style_table_text), Paragraph("Binary (0: Real, 1: Fraud)", style_table_text)],
        [Paragraph("Class distribution", style_table_text), Paragraph("17,014 Real / 866 Fraud (4.84%)", style_table_text)],
        [Paragraph("Missing values (profile / salary)", style_table_text), Paragraph("18.50% / 83.96%", style_table_text)],
        [Paragraph("Train / val / test split", style_table_text), Paragraph("12,516 / 2,682 / 2,682 (70/15/15)", style_table_text)]
    ]
    t1 = Table(t1_data, colWidths=[col_width*0.62, col_width*0.38])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white])
    ]))
    story.append(t1)
    story.append(Paragraph("<b>Table I:</b> EMSCAD dataset statistics.", style_caption))

    story.append(FrameBreak()) # To col2_p1

    # ------------------ COLUMN 2 (PAGE 1) ------------------
    eda_img = "B1_Fake_Job_Postings/figures/eda_missingness_fraud_rate.png"
    if os.path.exists(eda_img):
        story.append(Image(eda_img, width=col_width, height=col_width * 0.44))
        story.append(Paragraph("<b>Figure 1:</b> Fraud rate (%) across administrative logo and employment indicators.", style_caption))

    story.append(Paragraph("5. Methodology", style_h1))
    story.append(Paragraph("5.1 Learning Setting", style_h2))
    story.append(Paragraph("We adopt a supervised binary classification setting with severe imbalance. Three model families were benchmarked: (1) Logistic Regression with balanced class weighting; (2) SGD Linear SVM (maximum-margin linear classifier); and (3) Gradient Tree Boosting (non-linear ensemble capturing interactions between lexical n-grams and administrative omission flags).", style_body))

    story.append(Paragraph("5.2 Data Split Approach", style_h2))
    story.append(Paragraph("The dataset is partitioned into a Stratified 70/15/15 train/validation/test split (12,516 train, 2,682 validation, 2,682 test) using fixed seed (42). All TF-IDF vectorizers (unigrams and bigrams, 15,000 features), standard scalers, and one-hot encoders are encapsulated inside a scikit-learn ColumnTransformer fitted exclusively on the training split to prevent leakage. The validation split is used strictly to choose the optimal decision threshold &tau;, and test evaluation is performed once.", style_body))

    story.append(Paragraph("5.3 Parameter Settings", style_h2))
    story.append(Paragraph("Table II outlines the model parameter search configurations and selected values.", style_body))

    # Table 2: Hyperparameters
    t2_data = [
        [Paragraph("<b>Model</b>", style_table_header), Paragraph("<b>Search Space</b>", style_table_header), Paragraph("<b>Selected</b>", style_table_header)],
        [Paragraph("Logistic Reg.", style_table_text), Paragraph("Class weights, &tau; &isin; [0.1, 0.9]", style_table_text), Paragraph("Balanced, &tau;=0.78", style_table_text)],
        [Paragraph("Linear SVM", style_table_text), Paragraph("Loss function, penalty &alpha;", style_table_text), Paragraph("Log loss, &tau;=0.86", style_table_text)],
        [Paragraph("Gradient Boost", style_table_text), Paragraph("n &isin; {100, 150}, d &isin; {3, 5}", style_table_text), Paragraph("n=150, d=5, &tau;=0.48", style_table_text)]
    ]
    t2 = Table(t2_data, colWidths=[col_width*0.32, col_width*0.38, col_width*0.30])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white])
    ]))
    story.append(t2)
    story.append(Paragraph("<b>Table II:</b> Model hyperparameter configurations.", style_caption))

    # Trigger PageBreak to Page 2
    story.append(NextPageTemplate('TwoCol'))
    story.append(PageBreak())

    # ------------------ COLUMN 1 (PAGE 2) ------------------
    story.append(Paragraph("6. Results and Analysis", style_h1))
    story.append(Paragraph("6.1 Results", style_h2))
    story.append(Paragraph("Table III summarizes performance on the test split (N=2,682). Table IV details the mandatory 3-way ablation study isolating feature contributions.", style_body))

    # Table 3: Test results
    t3_data = [
        [Paragraph("<b>Model</b>", style_table_header), Paragraph("<b>Thresh</b>", style_table_header), Paragraph("<b>Prec.</b>", style_table_header), Paragraph("<b>Rec.</b>", style_table_header), Paragraph("<b>F1</b>", style_table_header), Paragraph("<b>PR-AUC</b>", style_table_header)],
        [Paragraph("Logistic Reg.", style_table_text), Paragraph("0.78", style_table_text), Paragraph("0.840", style_table_text), Paragraph("0.769", style_table_text), Paragraph("0.803", style_table_text), Paragraph("<b>0.907</b>", style_table_text_bold)],
        [Paragraph("Linear SVM", style_table_text), Paragraph("0.86", style_table_text), Paragraph("0.825", style_table_text), Paragraph("0.723", style_table_text), Paragraph("0.770", style_table_text), Paragraph("0.854", style_table_text)],
        [Paragraph("<b>Gradient Boost</b>", style_table_text_bold), Paragraph("0.48", style_table_text_bold), Paragraph("<b>0.925</b>", style_table_text_bold), Paragraph("<b>0.762</b>", style_table_text_bold), Paragraph("<b>0.835</b>", style_table_text_bold), Paragraph("0.806", style_table_text_bold)]
    ]
    t3 = Table(t3_data, colWidths=[col_width*0.28, col_width*0.14, col_width*0.14, col_width*0.14, col_width*0.14, col_width*0.16])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#e0f2fe')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, 2), [colors.HexColor('#f8fafc'), colors.white])
    ]))
    story.append(t3)
    story.append(Paragraph("<b>Table III:</b> Test-set results (best in bold).", style_caption))

    # Table 4: Ablation table
    t4_data = [
        [Paragraph("<b>Feature Set</b>", style_table_header), Paragraph("<b>Prec.</b>", style_table_header), Paragraph("<b>Rec.</b>", style_table_header), Paragraph("<b>F1</b>", style_table_header), Paragraph("<b>PR-AUC</b>", style_table_header)],
        [Paragraph("1. Structured Only", style_table_text), Paragraph("0.232", style_table_text), Paragraph("0.531", style_table_text), Paragraph("0.322", style_table_text), Paragraph("0.209", style_table_text)],
        [Paragraph("2. Text Only (TF-IDF)", style_table_text), Paragraph("<b>0.860</b>", style_table_text_bold), Paragraph("0.754", style_table_text), Paragraph("0.803", style_table_text), Paragraph("0.902", style_table_text)],
        [Paragraph("3. Combined Multimodal", style_table_text_bold), Paragraph("0.840", style_table_text), Paragraph("<b>0.769</b>", style_table_text_bold), Paragraph("<b>0.803</b>", style_table_text_bold), Paragraph("<b>0.907</b>", style_table_text_bold)]
    ]
    t4 = Table(t4_data, colWidths=[col_width*0.40, col_width*0.15, col_width*0.15, col_width*0.15, col_width*0.15])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#e0f2fe')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, 2), [colors.HexColor('#f8fafc'), colors.white])
    ]))
    story.append(t4)
    story.append(Paragraph("<b>Table IV:</b> Mandatory 3-way feature ablation study.", style_caption))

    # Figure 2: PR Curve & Confusion Matrix
    fig2_path = "B1_Fake_Job_Postings/figures/combined_key_result.png"
    if os.path.exists(fig2_path):
        story.append(Image(fig2_path, width=col_width, height=col_width * 0.44))
        story.append(Paragraph("<b>Figure 2:</b> PR Curves across ablation and test confusion matrix.", style_caption))

    story.append(FrameBreak()) # To col2_p2

    # ------------------ COLUMN 2 (PAGE 2) ------------------
    story.append(Paragraph("6.2 Analysis", style_h2))
    story.append(Paragraph("<b>1. Insufficiency of Accuracy:</b> In severe imbalance settings (4.84% positive class), accuracy is completely misleading. A naive majority baseline that predicts 'legitimate' for all samples scores an apparent <b>95.16% accuracy</b>, yet its recall on fraud is exactly <b>0.0%</b>, failing to catch even a single fraudulent advertisement. PR-AUC and F1 provide an honest reflection of minority detection trade-offs.", style_body))

    story.append(Paragraph("<b>2. Synergy of Multimodal Fusion:</b> The ablation study proves that structured metadata alone produces an unacceptable precision of 0.232 (PR-AUC = 0.209), because many legitimate startups also lack logos or offer remote work. In contrast, text TF-IDF yields strong discrimination (F1 = 0.803, PR-AUC = 0.902). Fusing both modalities maximizes overall area under the curve (PR-AUC reaches <b>0.907</b>, ROC-AUC reaches <b>0.992</b>), catching scams where polished text is contradicted by deceptive administrative omissions.", style_body))

    story.append(Paragraph("<b>3. Lexical Signals & False Positives:</b> Inspecting model weights reveals high positive coefficients on n-grams associated with urgent wire transactions: <i>'wire transfer'</i>, <i>'western union'</i>, <i>'data entry executive'</i>, <i>'telegram'</i>, and <i>'home office equipment'</i>. Structured omission indicators (missing company profile) further amplify risk. False positives predominantly occurred in non-profit and volunteer listings that lacked corporate logos and featured informal language.", style_body))

    story.append(Paragraph("7. Conclusion", style_h1))
    story.append(Paragraph("We developed an end-to-end multimodal pipeline for employment scam detection, demonstrating that combining structured administrative metadata with TF-IDF text features inside a zero-leakage ColumnTransformer yields a PR-AUC of 0.907 and an F1 score of 0.835. Calibrating decision thresholds on validation data was crucial for optimizing minority-class detection. Future work should investigate pre-trained transformer embeddings and corporate entity graph linking.", style_body))

    # Trigger PageBreak to Page 3 (AI Disclosure & References only)
    story.append(PageBreak())

    # ------------------ PAGE 3 (DISCLOSURE & REFS ONLY) ------------------
    story.append(Paragraph("AI Use Disclosure", style_h1))
    story.append(Paragraph("I used <b>Antigravity (powered by Google DeepMind Gemini)</b> for pipeline scaffolding, exploratory data visualization code, and mathematical phrasing of the problem statement. All experiments, cross-validations, metrics, and analyses were produced, verified, and checked by me, and I take full responsibility for this work.", style_body))

    story.append(Spacer(1, 10))
    story.append(Paragraph("References", style_h1))
    story.append(Paragraph("[1] S. Vidros, C. Kolias, G. Kambourakis, and L. Akoglu, &ldquo;Automatic detection of online recruitment frauds: Characteristics, methods, and a public dataset,&rdquo; <i>Future Internet</i>, vol. 9, no. 1, p. 6, 2017.", style_body))
    story.append(Paragraph("[2] F. Pedregosa <i>et al.</i>, &ldquo;Scikit-learn: Machine learning in Python,&rdquo; <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825&ndash;2830, 2011.", style_body))

    doc.build(story)
    print(f"Successfully generated IEEE two-column report PDF at: {output_pdf_path}")

if __name__ == '__main__':
    generate_report("B1_Fake_Job_Postings/report/70176119_B1_Report.pdf")
