"""
ReportLab IEEE 2-Column PDF Generator for Task B3 Report
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
    story.append(Paragraph("Task [B3]: [Predicting Star Ratings of Clothing Reviews]", style_title))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Muhammad Zaryab Attiq</b> &nbsp;|&nbsp; Roll No.: <b>70176119</b> &nbsp;|&nbsp; Section: <b>Section A</b><br/>"
        "Repository: <u>https://github.com/Vex456/AI101-ML-Assignment-70176119</u><br/>"
        "Instructor: <b>Dr. Muhammad Awais</b>, AI Architect", style_author))
    story.append(FrameBreak()) # To Col 1 Page 1

    # ------------------ PAGE 1 - COLUMN 1 ------------------
    story.append(Paragraph("I. INTRODUCTION", style_h1))
    story.append(Paragraph(
        "Customer reviews in online apparel retail provide indispensable feedback for conversion optimization and inventory management. In this work, we address the challenge of predicting ordinal 1-to-5 star ratings from 22,641 women's e-commerce clothing reviews by combining unstructured customer text with structured administrative metadata (customer age, positive feedback count, review word count, and product departments). To preserve scientific validity, we enforce the leakage rule by omitting the concurrent Recommended IND feature. We perform a three-way ablation study (structured-only vs. text-only vs. multimodal), benchmark class balancing methods, and analyze a re-framed 3-class commercial sentiment framing (Negative, Neutral, Positive), which elevates accuracy from 61.35% to 81.63%.",
        style_body
    ))

    story.append(Paragraph("II. PROBLEM STATEMENT", style_h1))
    story.append(Paragraph(
        "Given an input record <i>x = (x<sub>text</sub>, x<sub>num</sub>, x<sub>cat</sub>)</i>, the primary objective is to predict the discrete ordinal rating <i>y &isin; {1, 2, 3, 4, 5}</i>. Because the distribution exhibits severe positive skew (55.4% are 5-star reviews), overall accuracy is misleading. Evaluation relies on Macro-Averaged F1-Score: Macro-F1 = (1/5) &Sigma; F<sub>1,c</sub> and Ordinal Mean Absolute Error (MAE): (1/N) &Sigma; |y<sub>i</sub> - y^<sub>i</sub>|, measuring average star displacement.",
        style_body
    ))

    story.append(Paragraph("III. DATASET USED", style_h1))
    story.append(Paragraph(
        "We employ the Women's E-Commerce Clothing Reviews dataset from Kaggle. The cleaned corpus contains 22,641 valid reviews. Features comprise the review narrative, customer age, positive feedback count, division, department, and apparel class.",
        style_body
    ))

    story.append(Paragraph("IV. LEAKAGE RULE & STATISTICS", style_h1))
    story.append(Paragraph(
        "<b>Data Leakage Constraint:</b> The feature Recommended IND is strictly excluded. Because consumers tick the recommendation box concurrently with awarding star ratings, this feature is unavailable at inference time. Including it constitutes severe label leakage.<br/>"
        "Table I displays the class breakdown: 5 stars (55.4%), 4 stars (21.7%), 3 stars (12.5%), 2 stars (6.8%), and 1 star (3.6%). Dissatisfied reviewers write longer descriptions (median 68 words for 2-star vs. 52 words for 5-star).",
        style_body
    ))

    # Table I: Dataset Statistics
    table_data_stats = [
        [Paragraph("<b>Property</b>", style_table_header), Paragraph("<b>Value</b>", style_table_header)],
        [Paragraph("Total clean reviews", style_table_text_bold), Paragraph("22,641", style_table_text)],
        [Paragraph("Target scale", style_table_text), Paragraph("Ordinal [1-5 stars]", style_table_text)],
        [Paragraph("Class skew (5-star majority)", style_table_text_bold), Paragraph("12,540 (55.39%)", style_table_text_bold)],
        [Paragraph("Minority class (1-star)", style_table_text), Paragraph("821 (3.63%)", style_table_text)],
        [Paragraph("Excluded leakage feature", style_table_text_bold), Paragraph("Recommended IND", style_table_text_bold)],
        [Paragraph("Train / val / test split", style_table_text), Paragraph("15,848 / 3,396 / 3,397 (70/15/15)", style_table_text)],
        [Paragraph("Preprocessing", style_table_text), Paragraph("ColumnTransformer (TF-IDF + OneHot)", style_table_text)],
    ]
    t_stats = Table(table_data_stats, colWidths=[col_width * 0.62, col_width * 0.38])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(t_stats)
    story.append(Paragraph("TABLE I: Dataset statistics and star distribution.", style_caption))

    story.append(FrameBreak()) # Move to Col 2 Page 1

    # ------------------ PAGE 1 - COLUMN 2 ------------------
    if os.path.exists("B3_Clothing_Reviews/figures/eda_clothing_distributions.png"):
        story.append(Image("B3_Clothing_Reviews/figures/eda_clothing_distributions.png", width=col_width, height=col_width * 0.72))
        story.append(Paragraph("Fig. 1: (a) Rating counts; (b) Review length by rating; (c) Age density; (d) Rating distribution across departments.", style_caption))

    story.append(Paragraph("V. METHODOLOGY", style_h1))
    story.append(Paragraph("A. Data Partitioning & Pipeline", style_h2))
    story.append(Paragraph(
        "The dataset is split into a stratified 70/15/15 train/validation/test partition (seed=42). We structure three preprocessing pipelines: (1) Structured: numeric age, log(1 + feedback), review length scaled via StandardScaler; division, department, class encoded via OneHotEncoder; (2) Text: TF-IDF vectorizer (uni/bi-grams, max 5,000 features); (3) Multimodal: concatenation within a single ColumnTransformer.",
        style_body
    ))

    story.append(Paragraph("B. Model Portfolio & Re-Framing", style_h2))
    story.append(Paragraph(
        "We evaluate Logistic Regression (unweighted vs. balanced), Linear SVM, and Random Forest. We further train a re-framed 3-class classifier mapping ratings into Negative (1-2), Neutral (3), and Positive (4-5).",
        style_body
    ))

    story.append(NextPageTemplate('TwoCol'))
    story.append(PageBreak()) # Move to Page 2

    # ------------------ PAGE 2 - COLUMN 1 ------------------
    story.append(Paragraph("VI. RESULTS", style_h1))
    story.append(Paragraph(
        "Table II summarizes the 5-class ablation and 3-class results on the held-out test set (3,397 samples). Linear SVM achieved champion 5-class performance (Macro-F1 0.3958, Ordinal MAE 0.5131), while balanced weighting maximized Macro-F1 (0.4335). Re-framing into 3 commercial sentiment tiers boosted accuracy to 81.63% (Macro-F1 0.5540).",
        style_body
    ))

    # Table II: Results
    table_data_results = [
        [Paragraph("<b>Model / Modality</b>", style_table_header), Paragraph("<b>Acc</b>", style_table_header), Paragraph("<b>Macro-F1</b>", style_table_header), Paragraph("<b>W-F1</b>", style_table_header), Paragraph("<b>Ord. MAE</b>", style_table_header)],
        [Paragraph("Structured-Only", style_table_text), Paragraph("0.5540", style_table_text), Paragraph("0.1426", style_table_text), Paragraph("0.3950", style_table_text), Paragraph("0.8160", style_table_text)],
        [Paragraph("Text-Only", style_table_text), Paragraph("<b>0.6314</b>", style_table_text_bold), Paragraph("0.3830", style_table_text), Paragraph("0.5915", style_table_text), Paragraph("<b>0.5004</b>", style_table_text_bold)],
        [Paragraph("Combined (Unweighted)", style_table_text), Paragraph("0.6273", style_table_text), Paragraph("0.3756", style_table_text), Paragraph("0.5876", style_table_text), Paragraph("0.5110", style_table_text)],
        [Paragraph("Combined (Balanced)", style_table_text_bold), Paragraph("0.5890", style_table_text), Paragraph("<b>0.4335</b>", style_table_text_bold), Paragraph("<b>0.6011</b>", style_table_text_bold), Paragraph("0.5508", style_table_text)],
        [Paragraph("Combined (Linear SVM)", style_table_text), Paragraph("0.6135", style_table_text), Paragraph("0.3958", style_table_text), Paragraph("0.5896", style_table_text), Paragraph("0.5131", style_table_text)],
        [Paragraph("3-Class Re-Framed", style_table_text_bold), Paragraph("<b>0.8163</b>", style_table_text_bold), Paragraph("<b>0.5540</b>", style_table_text_bold), Paragraph("<b>0.7883</b>", style_table_text_bold), Paragraph("--", style_table_text)],
    ]
    t_res = Table(table_data_results, colWidths=[col_width * 0.44, col_width * 0.14, col_width * 0.14, col_width * 0.14, col_width * 0.14])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
    ]))
    story.append(t_res)
    story.append(Paragraph("TABLE II: Ablation study and 3-class results on test set.", style_caption))

    if os.path.exists("B3_Clothing_Reviews/figures/combined_key_result.png"):
        story.append(Image("B3_Clothing_Reviews/figures/combined_key_result.png", width=col_width, height=col_width * 0.44))
        story.append(Paragraph("Fig. 2: (a) 5x5 confusion matrix; (b) Macro-F1 across modalities and 3-class framing.", style_caption))

    story.append(Paragraph("VII. DISCUSSION & ANALYSIS", style_h1))
    story.append(Paragraph("A. Confusion Dynamics: 4 vs. 5 Star Ambiguity", style_h2))
    story.append(Paragraph(
        "The confusion matrix in Fig. 2(a) indicates that the largest error concentration occurs between 4 stars and 5 stars (over 480 4-star reviews predicted as 5 stars). This stems from subjective consumer label noise: two reviewers expressing identical satisfaction often diverge arbitrarily between 4 and 5 stars based on personal calibration. The low Ordinal MAE (0.513) confirms that nearly all errors are off-by-one.",
        style_body
    ))

    story.append(FrameBreak()) # Move to Col 2 Page 2

    # ------------------ PAGE 2 - COLUMN 2 ------------------
    story.append(Paragraph("B. Contribution of Structured Metadata", style_h2))
    story.append(Paragraph(
        "Structured administrative features alone yield a poor Macro-F1 of only 0.1426 (collapsing to majority-class prediction). Adding metadata to text produces virtually identical accuracy (0.6314 vs. 0.6273), demonstrating that customer review text contains overwhelmingly dominant predictive power.",
        style_body
    ))

    story.append(Paragraph("C. Business Utility of 3-Class Re-Framing", style_h2))
    story.append(Paragraph(
        "Fine-grained differentiation between 4 and 5 stars offers minimal utility for e-commerce inventory decisions. Grouping into Negative (1-2), Neutral (3), and Positive (4-5) allows automated triage of customer dissatisfaction, elevating operational classification accuracy to 81.63% (Macro-F1 0.5540).",
        style_body
    ))

    story.append(Paragraph("VIII. CONCLUSION", style_h1))
    story.append(Paragraph(
        "Multimodal text classification successfully predicts consumer clothing review ratings. Excluding Recommended IND was critical to prevent label leakage. Class weighting improved minority Macro-F1, and 3-class reframing maximized operational business value.",
        style_body
    ))

    story.append(PageBreak()) # Move to Page 3 (Disclosures & References)

    # ------------------ PAGE 3 - DISCLOSURE & REFS ------------------
    story.append(Paragraph("AI USE DISCLOSURE STATEMENT", style_h1))
    story.append(Paragraph(
        "• <b>AI Tool Utilized:</b> Google Antigravity Agent (Gemini 2.5 Pro).<br/>"
        "• <b>Scope of Assistance:</b> Scaffolding multimodal ColumnTransformers, ordinal classification metrics, and IEEE two-column report compilation.<br/>"
        "• <b>Human Verification:</b> All models and reframing evaluations were executed locally with fixed random seeds (seed=42); all reported numbers match notebook outputs 1:1.",
        style_body
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("REFERENCES", style_h1))
    refs = [
        "[1] S. Nicapotato, 'Women's E-Commerce Clothing Reviews,' <i>Kaggle Dataset</i>, 2018.",
        "[2] F. Pedregosa <i>et al.</i>, 'Scikit-learn: Machine learning in Python,' <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825-2830, 2011."
    ]
    for ref in refs:
        story.append(Paragraph(ref, style_body))

    doc.build(story)
    print(f"Report successfully compiled to: {output_pdf_path}")

if __name__ == '__main__':
    generate_report("B3_Clothing_Reviews/report/70176119_B3_Report.pdf")
