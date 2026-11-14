"""
ReportLab IEEE 2-Column PDF Generator for Task A2 Report
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
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

def generate_report(output_pdf_path):
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    # Page dimensions (A4: 595.27 x 841.89 points)
    page_width, page_height = A4
    margin_x = 36  # 0.5 in = 36 pt
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
    style_body_lead = ParagraphStyle('BodyLead', fontName='Times-Italic', fontSize=8.2, leading=10.2, alignment=TA_JUSTIFY, spaceAfter=4, textColor=colors.HexColor('#0f172a'))
    style_table_text = ParagraphStyle('TableText', fontName='Times-Roman', fontSize=7.2, leading=8.5, alignment=TA_CENTER)
    style_table_text_bold = ParagraphStyle('TableTextBold', fontName='Helvetica-Bold', fontSize=7.2, leading=8.5, alignment=TA_CENTER)
    style_table_header = ParagraphStyle('TableHeader', fontName='Helvetica-Bold', fontSize=7.5, leading=9, alignment=TA_CENTER, textColor=colors.white)
    style_caption = ParagraphStyle('Caption', fontName='Helvetica-Oblique', fontSize=7.2, leading=8.5, alignment=TA_CENTER, textColor=colors.HexColor('#334155'), spaceBefore=2, spaceAfter=4)

    story = []

    # ------------------ FIRST PAGE HEADER ------------------
    story.append(Paragraph("THE UNIVERSITY OF LAHORE • DEPARTMENT OF INTELLIGENT SYSTEMS • AI-101", style_supertitle))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Task [A2]: Predicting Concrete Compressive Strength", style_title))
    story.append(Spacer(1, 3))
    story.append(Paragraph("<b>Muhammad Zaryab Attiq</b> &nbsp;|&nbsp; Roll No. 70176119 &nbsp;|&nbsp; Section A<br/>Repository: <u>https://github.com/Muhammad-Zaryab-Attiq/AI101-ML-Assignment-70176119</u> &nbsp;|&nbsp; Instructor: Dr. Muhammad Awais, AI Architect", style_author))
    story.append(FrameBreak()) # End of header frame, flow to col1_p1

    # ------------------ COLUMN 1 (PAGE 1) ------------------
    story.append(Paragraph("1. Introduction", style_h1))
    story.append(Paragraph("Testing laboratory concrete compressive strength via traditional hydraulic crushing cylinders requires up to 28 days of curing in water baths, creating severe bottlenecks in modern construction timelines and quality assurance workflows. In this investigation, we formulate a supervised regression problem to estimate the 28-day compressive strength (in MegaPascals, MPa) directly from raw constituent mix proportions and curing age. By augmenting the feature space with physical chemistry formulations—specifically log(Age) to linearize hydration kinetics and the water-to-cement ratio following Abrams' Law—gradient boosting regression achieved a test-set RMSE of <b>4.555 MPa</b> and an R² of <b>0.919</b>, outperforming the un-engineered linear regression baseline by over 53% in error reduction.", style_body))

    story.append(Paragraph("2. Problem Statement", style_h1))
    story.append(Paragraph("Given an input vector of mix design components and sample maturity <i>x</i> &isin; &real;<sup><i>d</i></sup> (<i>d</i>=8 for raw constituents, <i>d</i>=10 with engineered physical ratios), our objective is to learn a hypothesis function <i>f: x &mapsto; y</i> that predicts concrete compressive strength <i>y</i> &isin; [2.33, 82.60] MPa. Success is evaluated using Root Mean Squared Error (RMSE) as the primary metric to heavily penalize large structural safety deviations, alongside Mean Absolute Error (MAE) and the coefficient of determination (R²). The central physical constraint is the pronounced non-linearity of cementitious hydration, wherein strength accelerates steeply during early curing before plateauing logarithmically, severely violating standard Gauss-Markov linearity assumptions.", style_body))

    story.append(Paragraph("3. Dataset Used", style_h1))
    story.append(Paragraph("We employ the Concrete Compressive Strength dataset donated by Prof. I-Cheng Yeh to the UCI Machine Learning Repository (Dataset ID: 165) [1]. The dataset comprises 1,030 experimental concrete formulations. Each record consists of seven constituent ingredient masses measured in kg/m³ (Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate) and the specimen curing age in days (1 to 365). Inspection confirms zero missing values and zero duplicate records across all 8 input columns. All numerical inputs were standardized using training-set statistics within a scikit-learn Pipeline [2] to guarantee zero data leakage.", style_body))

    story.append(Paragraph("4. Dataset Statistics", style_h1))
    story.append(Paragraph("Table I summarizes key dataset properties. Target compressive strength spans 2.33 to 82.60 MPa with a mean of 35.82 &plusmn; 16.71 MPa. Curing age exhibits severe positive skewness with a median of 28 days and a maximum of 365 days.", style_body))

    # Table 1: Dataset Statistics
    t1_data = [
        [Paragraph("<b>Property</b>", style_table_header), Paragraph("<b>Value</b>", style_table_header)],
        [Paragraph("Samples (after cleaning)", style_table_text), Paragraph("1,030", style_table_text)],
        [Paragraph("Features (num / cat / text)", style_table_text), Paragraph("8 / 0 / 0", style_table_text)],
        [Paragraph("Target range", style_table_text), Paragraph("[2.33, 82.60] MPa", style_table_text)],
        [Paragraph("Target mean &plusmn; sd", style_table_text), Paragraph("35.82 &plusmn; 16.71 MPa", style_table_text)],
        [Paragraph("Missing values (%)", style_table_text), Paragraph("0.0%", style_table_text)],
        [Paragraph("Train / test split", style_table_text), Paragraph("824 / 206 (80 / 20)", style_table_text)]
    ]
    t1 = Table(t1_data, colWidths=[col_width*0.62, col_width*0.38])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white])
    ]))
    story.append(t1)
    story.append(Paragraph("<b>Table I:</b> Concrete compressive strength dataset statistics.", style_caption))

    story.append(FrameBreak()) # End of col1_p1, flow to col2_p1

    # ------------------ COLUMN 2 (PAGE 1) ------------------
    eda_img_path = "A2_Concrete_Strength/figures/eda_correlation_matrix.png"
    if os.path.exists(eda_img_path):
        img_eda = Image(eda_img_path, width=col_width, height=col_width * 0.78)
        story.append(img_eda)
        story.append(Paragraph("<b>Figure 1:</b> Correlation matrix of mix constituents, age, and compressive strength.", style_caption))

    story.append(Paragraph("5. Methodology", style_h1))
    story.append(Paragraph("5.1 Learning Setting", style_h2))
    story.append(Paragraph("We formulate a supervised regression task comparing four distinct model families: (1) Ordinary Least Squares Linear Regression (baseline); (2) Polynomial Features with Ridge Regularization (captures 2nd-order ingredient interactions under L2 penalty); (3) Random Forest and Gradient Tree Boosting (non-parametric ensembles capturing non-linear partitions); and (4) Multi-Layer Perceptron (MLPRegressor with ReLU activations and Adam optimizer).", style_body))

    story.append(Paragraph("5.2 Data Split Approach", style_h2))
    story.append(Paragraph("The dataset of 1,030 samples is split into an 80% training set (824 samples) and a 20% test set (206 samples) with fixed seed (42). All scalers (StandardScaler) are enclosed in scikit-learn Pipeline objects fitted strictly on training data to prevent leakage. Hyperparameter tuning is executed via 5-fold cross-validation exclusively on the training split.", style_body))

    story.append(Paragraph("5.3 Parameter Settings", style_h2))
    story.append(Paragraph("Table II displays the hyperparameter grid searches and optimal configurations selected via 5-fold cross-validation.", style_body))

    # Table 2: Hyperparameters
    t2_data = [
        [Paragraph("<b>Model</b>", style_table_header), Paragraph("<b>Search Space</b>", style_table_header), Paragraph("<b>Selected</b>", style_table_header)],
        [Paragraph("Linear Reg.", style_table_text), Paragraph("None (OLS)", style_table_text), Paragraph("Default", style_table_text)],
        [Paragraph("Poly + Ridge", style_table_text), Paragraph("&alpha; &isin; {0.1, 1, 10, 100}", style_table_text), Paragraph("deg=2, &alpha;=10.0", style_table_text)],
        [Paragraph("Random Forest", style_table_text), Paragraph("n &isin; {100, 200}, d &isin; {10, 15}", style_table_text), Paragraph("n=200, d=15", style_table_text)],
        [Paragraph("Gradient Boost", style_table_text), Paragraph("n &isin; {100, 200}, lr &isin; {0.05, 0.1}", style_table_text), Paragraph("n=200, lr=0.1, d=5", style_table_text)],
        [Paragraph("MLPRegressor", style_table_text), Paragraph("Architecture, &alpha;", style_table_text), Paragraph("(64, 32), &alpha;=0.01", style_table_text)]
    ]
    t2 = Table(t2_data, colWidths=[col_width*0.32, col_width*0.38, col_width*0.30])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white])
    ]))
    story.append(t2)
    story.append(Paragraph("<b>Table II:</b> Model hyperparameter search space and optimal values.", style_caption))

    # Trigger PageBreak to Page 2
    story.append(NextPageTemplate('TwoCol'))
    story.append(PageBreak())

    # ------------------ COLUMN 1 (PAGE 2) ------------------
    story.append(Paragraph("6. Results and Analysis", style_h1))
    story.append(Paragraph("6.1 Results", style_h2))
    story.append(Paragraph("All models were evaluated on the identical held-out test set (206 instances). Table III presents the comparative results across raw and engineered feature representations.", style_body))

    # Table 3: Results
    t3_data = [
        [Paragraph("<b>Model</b>", style_table_header), Paragraph("<b>Features</b>", style_table_header), Paragraph("<b>MAE</b>", style_table_header), Paragraph("<b>RMSE</b>", style_table_header), Paragraph("<b>R²</b>", style_table_header)],
        [Paragraph("Linear Reg.", style_table_text), Paragraph("Raw (8)", style_table_text), Paragraph("7.746", style_table_text), Paragraph("9.796", style_table_text), Paragraph("0.628", style_table_text)],
        [Paragraph("Linear Reg.", style_table_text), Paragraph("Engineered", style_table_text), Paragraph("5.295", style_table_text), Paragraph("6.562", style_table_text), Paragraph("0.833", style_table_text)],
        [Paragraph("Poly + Ridge", style_table_text), Paragraph("Raw (8)", style_table_text), Paragraph("5.963", style_table_text), Paragraph("7.430", style_table_text), Paragraph("0.786", style_table_text)],
        [Paragraph("Poly + Ridge", style_table_text), Paragraph("Engineered", style_table_text), Paragraph("4.358", style_table_text), Paragraph("5.483", style_table_text), Paragraph("0.883", style_table_text)],
        [Paragraph("Random Forest", style_table_text), Paragraph("Raw (8)", style_table_text), Paragraph("3.782", style_table_text), Paragraph("5.539", style_table_text), Paragraph("0.881", style_table_text)],
        [Paragraph("Random Forest", style_table_text), Paragraph("Engineered", style_table_text), Paragraph("3.694", style_table_text), Paragraph("5.356", style_table_text), Paragraph("0.889", style_table_text)],
        [Paragraph("<b>Gradient Boost</b>", style_table_text_bold), Paragraph("<b>Engineered</b>", style_table_text_bold), Paragraph("<b>2.917</b>", style_table_text_bold), Paragraph("<b>4.555</b>", style_table_text_bold), Paragraph("<b>0.919</b>", style_table_text_bold)],
        [Paragraph("MLPRegressor", style_table_text), Paragraph("Engineered", style_table_text), Paragraph("3.554", style_table_text), Paragraph("5.060", style_table_text), Paragraph("0.901", style_table_text)]
    ]
    t3 = Table(t3_data, colWidths=[col_width*0.27, col_width*0.25, col_width*0.16, col_width*0.16, col_width*0.16])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('BACKGROUND', (0, 7), (-1, 7), colors.HexColor('#e0f2fe')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, 6), [colors.HexColor('#f8fafc'), colors.white]),
        ('ROWBACKGROUNDS', (0, 8), (-1, 8), [colors.white])
    ]))
    story.append(t3)
    story.append(Paragraph("<b>Table III:</b> Test-set evaluation results (best performance in bold).", style_caption))

    # Figure 2: Combined key result
    fig2_path = "A2_Concrete_Strength/figures/combined_key_result.png"
    if os.path.exists(fig2_path):
        img_fig2 = Image(fig2_path, width=col_width, height=col_width * 0.45)
        story.append(img_fig2)
        story.append(Paragraph("<b>Figure 2:</b> Predicted vs. actual strength and residual distribution for champion GBDT.", style_caption))

    story.append(Paragraph("6.2 Analysis", style_h2))
    story.append(Paragraph("<b>1. Baseline Under-performance & Linearity Violation:</b> The linear regression baseline achieves poor accuracy on raw attributes (R² = 0.628, RMSE = 9.796 MPa) because cementitious hydration violates the Gauss-Markov linearity assumption. Concrete strength exhibits asymptotic growth: strength surges rapidly during early curing (&le; 14 days) and subsequently plateaus logarithmically towards 365 days. A standard linear hypothesis imposes a constant rate of change across the curing horizon, severely overestimating green concrete while underestimating mature samples.", style_body))

    story.append(FrameBreak()) # End of col1_p2, flow to col2_p2

    # ------------------ COLUMN 2 (PAGE 2) ------------------
    story.append(Paragraph("<b>2. Differential Impact of Feature Engineering:</b> Augmenting the feature matrix with log(Age) and the water-to-cement ratio (w/c) yields dramatic gains for linear models: Linear Regression RMSE plummeted by 33% (from 9.796 to 6.562 MPa, raising R² to 0.833), and Polynomial Ridge R² surged from 0.786 to 0.883. Linearizing the temporal curing axis allows linear hyperplanes to fit the logarithmic kinetic profile directly. Conversely, tree-based ensembles (Random Forest and Gradient Boosting) partition continuous feature space via axis-aligned orthogonal step functions, naturally approximating non-linear curvatures even on raw features (RF raw R² = 0.881). Nonetheless, the explicit w/c feature consolidates two collinear splits into a single physical governing metric, refining Gradient Boosting to a champion test RMSE of <b>4.555 MPa</b> (R² = <b>0.919</b>).", style_body))

    story.append(Paragraph("<b>3. Residual Diagnostics & Engineering Insights:</b> Residual analysis reveals that the largest prediction errors occur in early curing stages (&le; 7 days), where initial hydration kinetics are volatile and highly sensitive to unmeasured ambient temperature, and in high-strength mixes (&gt; 65 MPa) with elevated blast-furnace slag content. Feature importances confirm that Curing Age, Cement, and the w/c ratio account for over 72% of tree splits, corroborating Abrams' classical law in civil engineering.", style_body))

    story.append(Paragraph("7. Conclusion", style_h1))
    story.append(Paragraph("This study demonstrates that marrying domain-grounded physical features (log(Age) and water/cement ratio) with Gradient Tree Boosting produces an accurate predictive model for concrete compressive strength (R² = 0.919, RMSE = 4.555 MPa). Linear models were significantly improved through physical transformations. Future work should integrate ambient curing temperature telemetry and explore physics-informed neural loss functions.", style_body))

    # Trigger PageBreak to Page 3 (AI Disclosure & References only)
    story.append(PageBreak())

    # ------------------ PAGE 3 (DISCLOSURE & REFS ONLY) ------------------
    story.append(Paragraph("AI Use Disclosure", style_h1))
    story.append(Paragraph("I used <b>Antigravity (powered by Google DeepMind Gemini)</b> for pipeline scaffolding, exploratory data visualization code, and mathematical phrasing of the problem statement. All experiments, cross-validations, metrics, and analyses were produced, verified, and checked by me, and I take full responsibility for this work.", style_body))

    story.append(Spacer(1, 10))
    story.append(Paragraph("References", style_h1))
    story.append(Paragraph("[1] I-C. Yeh, &ldquo;Modeling of strength of high-performance concrete using artificial neural networks,&rdquo; <i>Cement and Concrete Research</i>, vol. 28, no. 12, pp. 1797&ndash;1808, 1998.", style_body))
    story.append(Paragraph("[2] F. Pedregosa <i>et al.</i>, &ldquo;Scikit-learn: Machine learning in Python,&rdquo; <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825&ndash;2830, 2011.", style_body))

    doc.build(story)
    print(f"Successfully generated IEEE two-column report PDF at: {output_pdf_path}")

if __name__ == '__main__':
    generate_report("A2_Concrete_Strength/report/70176119_A2_Report.pdf")
