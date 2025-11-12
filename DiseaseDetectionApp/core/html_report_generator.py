



import datetime
import os
import base64

def get_domain_logo(domain):
    """
    Placeholder for domain-specific logo. Returns None as no logos are implemented.
    """
    return None

def generate_html_report(diagnosis_data, file_path):
    """
    Generates an HTML report from diagnosis data and saves it to file_path.

    Args:
        diagnosis_data (dict): Dictionary containing diagnosis information.
        file_path (str): Path to save the HTML file.

    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    try:
        domain = diagnosis_data.get('domain', 'general').lower()
        domain_title = f"{domain.capitalize()} Disease Diagnosis Report" if domain != 'general' else "Disease Diagnosis Report"


        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{domain_title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f4f4f4;
            color: #333;
        }}
        .header {{
            text-align: center;
            background-color: #007bff;
            color: white;
            padding: 20px;
            border-radius: 10px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .summary {{
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .summary table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .summary th, .summary td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .summary th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .section {{
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #007bff;
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .stages {{
            background-color: #e9ecef;
            padding: 10px;
            border-radius: 5px;
        }}
        .solution {{
            background-color: #d4edda;
            padding: 10px;
            border-radius: 5px;
            color: #155724;
            font-weight: bold;
        }}
        .disclaimer {{
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-style: italic;
        }}
        .footer {{
            text-align: center;
            margin-top: 20px;
            font-size: 12px;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{domain_title}</h1>
        <p>Report Generated on: {datetime.date.today().strftime('%B %d, %Y')}</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr><th>Domain:</th><td>{diagnosis_data.get('domain', 'N/A')}</td></tr>
            <tr><th>Disease Name:</th><td><strong>{diagnosis_data.get('name', 'N/A')}</strong></td></tr>
            <tr><th>Confidence Score:</th><td>{diagnosis_data.get('confidence', 'N/A')}</td></tr>
            <tr><th>Predicted Stage:</th><td>{diagnosis_data.get('stage', 'N/A')}</td></tr>
        </table>
    </div>
"""


        image_path = diagnosis_data.get('image_path')
        if image_path and os.path.exists(image_path):

            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            image_ext = os.path.splitext(image_path)[1].lower()
            if image_ext == '.png':
                img_src = f"data:image/png;base64,{encoded_string}"
            elif image_ext in ['.jpg', '.jpeg']:
                img_src = f"data:image/jpeg;base64,{encoded_string}"
            else:
                img_src = None
            if img_src:
                html_content += f"""
    <div class="section">
        <h2>Image Provided for Analysis</h2>
        <div class="image-container">
            <img src="{img_src}" alt="Diagnosis Image">
        </div>
    </div>
"""
            else:
                html_content += f"""
    <div class="section">
        <h2>Image Provided for Analysis</h2>
        <p>Image could not be embedded in the report.</p>
    </div>
"""


        description = diagnosis_data.get('description', 'Not Available')
        if description and description != 'Not Available':
            description_html = description.replace('\n', '<br>')
            html_content += f"""
    <div class="section">
        <h2>Detailed Description</h2>
        <p>{description_html}</p>
    </div>
"""


        stages = diagnosis_data.get("stages")
        if stages and isinstance(stages, dict):
            stages_html = "<br>".join([f"<strong>{k}:</strong> {v}" for k, v in stages.items()])
            html_content += f"""
    <div class="section">
        <h2>Disease Progression Stages</h2>
        <div class="stages">
            {stages_html}
        </div>
    </div>
"""


        causes = diagnosis_data.get('causes', 'Not Available')
        if causes and causes != 'Not Available':
            html_content += f"""
    <div class="section">
        <h2>Common Causes</h2>
        <p>{causes.replace('\n', '<br>')}</p>
    </div>
"""


        preventive = diagnosis_data.get('preventive_measures', 'Not Available')
        if preventive and preventive != 'Not Available':
            html_content += f"""
    <div class="section">
        <h2>Preventive Measures</h2>
        <p>{preventive.replace('\n', '<br>')}</p>
    </div>
"""


        wiki_summary = diagnosis_data.get('wiki_summary', 'Not Available')
        if wiki_summary and wiki_summary != 'Not Available':
            html_content += f"""
    <div class="section">
        <h2>Wikipedia Summary</h2>
        <p>{wiki_summary.replace('\n', '<br>')}</p>
    </div>
"""


        pubmed_summary = diagnosis_data.get('pubmed_summary', 'Not Available')
        if pubmed_summary and pubmed_summary != 'Not Available':
            html_content += f"""
    <div class="section">
        <h2>Recent Research from PubMed</h2>
        <div>{pubmed_summary}</div>
    </div>
"""


        solution = diagnosis_data.get('solution', 'No specific solution provided in the database.')
        html_content += f"""
    <div class="section">
        <h2>Recommended Solution / Cure</h2>
        <div class="solution">
            {solution.replace('\n', '<br>')}
        </div>
    </div>
"""


        html_content += f"""
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This report is generated by an automated system and should be used for informational purposes only. Consult a qualified professional for a definitive diagnosis.
    </div>

    <div class="footer">
        <p>Generated by Multi-Species Disease Detection and Management System</p>
    </div>
</body>
</html>
"""


        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return True, None

    except Exception as e:
        return False, str(e)
