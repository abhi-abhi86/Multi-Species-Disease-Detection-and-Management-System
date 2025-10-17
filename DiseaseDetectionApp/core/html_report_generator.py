# DiseaseDetectionApp/core/html_report_generator.py
# HTML Report Generator for Disease Diagnosis
# Generates HTML reports from diagnosis data, with embedded images for portability.

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

        # HTML Template
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
            background-color: #f4f7fc;
            color: #2c3e50;
        }}
        .header {{
            text-align: center;
            background-color: #1d2c5e;
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
            background-color: #f4f7fc;
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
            color: #4a6baf;
            border-bottom: 2px solid #4a6baf;
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
            background-color: #e9f5e9;
            padding: 10px;
            border-radius: 5px;
        }}
        .solution {{
            background-color: #ffeaa7;
            padding: 10px;
            border-radius: 5px;
            color: #d63031;
            font-weight: bold;
        }}
        .disclaimer {{
            background-color: #fab1a0;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-style: italic;
        }}
        .footer {{
            text-align: center;
            margin-top: 20px;
            font-size: 12px;
            color: #7f8c8d;
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

        # Image Section
        image_path = diagnosis_data.get('image_path')
        if image_path and os.path.exists(image_path):
            # Encode image to base64 for embedding
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

        # Description
        description = diagnosis_data.get('description', 'Not Available')
        if description and description != 'Not Available':
            html_content += f"""
    <div class="section">
        <h2>Detailed Description</h2>
        <p>{description.replace('\n', '<br>')}</p>
    </div>
"""

        # Stages
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

        # Causes
        causes = diagnosis_data.get('causes', 'Not Available')
        if causes and causes != 'Not Available':
            html_content += f"""
    <div class="section">
        <h2>Common Causes</h2>
        <p>{causes.replace('\n', '<br>')}</p>
    </div>
"""

        # Preventive Measures
        preventive = diagnosis_data.get('preventive_measures', 'Not Available')
        if preventive and preventive != 'Not Available':
            html_content += f"""
    <div class="section">
        <h2>Preventive Measures</h2>
        <p>{preventive.replace('\n', '<br>')}</p>
    </div>
"""

        # Solution
        solution = diagnosis_data.get('solution', 'No specific solution provided in the database.')
        html_content += f"""
    <div class="section">
        <h2>Recommended Solution / Cure</h2>
        <div class="solution">
            {solution.replace('\n', '<br>')}
        </div>
    </div>
"""

        # Disclaimer
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

        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return True, None

    except Exception as e:
        return False, str(e)
