# DiseaseDetectionApp/core/report_generator.py
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.charts.axes import YCategoryAxis
import os

def on_page(canvas, doc):
    canvas.saveState()

    # --- Watermark ---
    watermark_text = "abhi-abhi86"
    canvas.setFont("Helvetica-Bold", 42)
    # For transparency, use setFillAlpha if available (ReportLab >=3.5), else use a light color
    if hasattr(canvas, "setFillAlpha"):
        canvas.setFillAlpha(0.12)
        canvas.setFillColorRGB(0.6, 0.6, 0.6)
    else:
        canvas.setFillColorRGB(0.85, 0.85, 0.85)
    canvas.saveState()
    canvas.translate(doc.pagesize[0] / 2, doc.pagesize[1] / 2)
    canvas.rotate(30)
    canvas.drawCentredString(0, 0, watermark_text)
    canvas.restoreState()
    if hasattr(canvas, "setFillAlpha"):
        canvas.setFillAlpha(1.0)

    # Header
    header_text = "Multi-Species Disease Diagnosis System"
    canvas.setFont('Helvetica-Bold', 12)
    canvas.setFillColorRGB(0, 0, 0)
    canvas.drawCentredString(letter[0] / 2.0, letter[1] - 0.5 * inch, header_text)
    canvas.line(0.5 * inch, letter[1] - 0.65 * inch, letter[0] - 0.5 * inch, letter[1] - 0.65 * inch)

    # Footer
    footer_text = f"Report Generated on: {datetime.date.today().strftime('%B %d, %Y')}"
    canvas.setFont('Helvetica', 9)
    canvas.drawString(0.5 * inch, 0.5 * inch, footer_text)
    canvas.drawRightString(letter[0] - 0.5 * inch, 0.5 * inch, f"Page {doc.page}")

    canvas.restoreState()

def create_stages_chart(stages_dict):
    if not stages_dict or len(stages_dict) < 2:
        return None

    drawing = Drawing(width=400, height=len(stages_dict) * 35 + 20)
    data = [[i+1] for i in range(len(stages_dict))]
    chart = HorizontalBarChart()
    chart.x = 60
    chart.y = 20
    chart.height = len(stages_dict) * 35
    chart.width = 320
    chart.data = data
    chart.valueAxis.visible = False
    chart.categoryAxis = YCategoryAxis()
    chart.categoryAxis.categoryNames = list(stages_dict.keys())
    chart.categoryAxis.labels.boxAnchor = 'e'
    chart.categoryAxis.labels.dx = -10
    chart.categoryAxis.labels.fontName = 'Helvetica'

    for i in range(len(data)):
        # Use colors.Color() for RGBA-style transparency
        color = colors.Color(74/255, 107/255, 175/255, alpha=0.4 + (i / len(data) * 0.6))
        chart.bars[i].fillColor = color
        chart.bars[i].strokeColor = colors.white

    drawing.add(chart)
    return drawing

def generate_pdf_report(file_path, diagnosis_data):
    try:
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle', parent=styles['Heading1'], fontSize=24, leading=30, alignment=TA_CENTER,
            spaceAfter=24, textColor=colors.HexColor('#1D2C5E')
        )
        header_style = ParagraphStyle(
            'HeaderStyle', parent=styles['Heading2'], fontSize=16, leading=20, spaceBefore=18,
            spaceAfter=8, textColor=colors.HexColor('#4A6BAF')
        )
        body_style = ParagraphStyle(
            'BodyStyle', parent=styles['Normal'], fontSize=11, leading=15,
            alignment=TA_JUSTIFY, spaceAfter=12
        )
        solution_style = ParagraphStyle(
            'SolutionStyle', parent=body_style, textColor=colors.darkgreen, backColor=colors.HexColor('#E9F5E9')
        )

        story = []
        story.append(Paragraph("Disease Diagnosis Report", title_style))
        story.append(Spacer(1, 0.1 * inch))

        summary_data = [
            [Paragraph('<b>Domain:</b>', body_style), Paragraph(diagnosis_data.get('domain', 'N/A'), body_style)],
            [Paragraph('<b>Disease Name:</b>', body_style), Paragraph(f"<b>{diagnosis_data.get('name', 'N/A')}</b>", body_style)],
            [Paragraph('<b>Confidence Score:</b>', body_style), Paragraph(f"{diagnosis_data.get('confidence', 0.0):.1f}%", body_style)],
            [Paragraph('<b>Predicted Stage:</b>', body_style), Paragraph(diagnosis_data.get('stage', 'N/A'), body_style)],
        ]
        summary_table = Table(summary_data, colWidths=[1.5 * inch, 5.5 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F4F7FC')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#D9E2F3')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3 * inch))

        image_path = diagnosis_data.get('image_path')
        if image_path and os.path.exists(image_path):
            story.append(Paragraph("Image Provided for Analysis", header_style))
            try:
                img = Image(image_path, width=3*inch, height=3*inch)
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Spacer(1, 0.3 * inch))
            except Exception as e:
                print(f"Could not include image in PDF: {e}")
                story.append(Paragraph("<i>[Image could not be loaded into report]</i>", body_style))

        def add_section(title, key):
            content = diagnosis_data.get(key, 'Not Available')
            if content and content.strip() and content != 'Not Available':
                story.append(Paragraph(title, header_style))
                story.append(Paragraph(content.replace('\n', '<br/>'), body_style))

        add_section("Detailed Description", "description")
        stages = diagnosis_data.get("stages")
        if stages:
            story.append(Paragraph("Disease Progression Stages", header_style))
            stages_text = "<br/>".join([f"<b>• {k}:</b> {v}" for k, v in stages.items()])
            story.append(Paragraph(stages_text, body_style))
            stages_chart = create_stages_chart(stages)
            if stages_chart:
                story.append(Spacer(1, 0.2 * inch))
                story.append(stages_chart)

        add_section("Common Causes", "causes")
        add_section("Preventive Measures", "preventive_measures")

        story.append(Paragraph("Recommended Solution / Cure", header_style))
        solution_text = diagnosis_data.get('solution', 'No specific solution provided in the database.')
        story.append(Paragraph(solution_text, solution_style))
        story.append(Spacer(1, 0.2 * inch))

        disclaimer_style = ParagraphStyle('disclaimer', parent=styles['Italic'], fontSize=9, alignment=TA_CENTER)
        story.append(Paragraph(
            "<i><b>Disclaimer:</b> This report is generated by an automated system and should be used for informational purposes only. "
            "Consult a qualified professional for a definitive diagnosis.</i>", disclaimer_style))

        doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
        return True, None

    except Exception as e:
        print(f"An error occurred while generating the PDF: {e}")
        return False, str(e)
