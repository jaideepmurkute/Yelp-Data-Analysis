
import os
import io
import re
import sys

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import Paragraph, Spacer, Image, HRFlowable
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from CFG import Config


# Create a SimpleDocTemplate object
config_dict = Config().get_config()
pdf_save_path = os.path.join(config_dict['output_dir'], 'report.pdf')
pdf = SimpleDocTemplate(pdf_save_path, pagesize=A4)

# Get the styles for the document
styles = getSampleStyleSheet()
styleN = styles["BodyText"]
styleH = styles["Heading1"]

# Content for the PDF
content = []

# -------------------------------------------------------------
# Add a title
# title = Paragraph("Monthly Sales Report", styleH)
# title = Paragraph("User Review Insights", styleH)
center_aligned_style = ParagraphStyle(
    name='CenterAligned',
    parent=styleH,
    alignment=TA_CENTER
)
content.append(Paragraph("User Review Insights", center_aligned_style))

# -------------------------------------------------------------

# add Subtitle - Business ID, Business Name, City, Category
bus_id = '12345'
bus_name = 'Test Business'
city = 'Test City'
category = 'Test Category'
subtitle = f"""
<b>Business ID:</b> {bus_id} <br/>
<b>Name:</b> {bus_name} <br/>
<b>City:</b> {city} <br/>
<b>Categories:</b> {category}
"""
# Define a left-aligned style for the subtitle
left_aligned_style = ParagraphStyle(
    name='LeftAligned',
    parent=styleN,
    alignment=TA_LEFT
)
content.append(Paragraph(subtitle, left_aligned_style))

# -------------------------------------------------------------

# Add some space
content.append(Spacer(1, 12))

# -------------------------------------------------------------

# Add a paragraph
text = """This report contains the actionable insights generated from the user reviews of the business."""
content.append(Paragraph(text, styleN))

# -------------------------------------------------------------

# Add some space
content.append(Spacer(1, 12))

# -------------------------------------------------------------
# -------------------------------------------------------------
# Add a paragraph
bus_rev_summary = """Business' review summary: """
bus_pos_summary_data = f"""
    <b> {bus_rev_summary} </b><br/>
    """
content.append(Paragraph(bus_pos_summary_data))

# ------------------------------------------------------------
# Add a paragraph
bus_pos_summary = """This is bussiness' positive review summary."""
bus_pos_summary_data = f"""
    &nbsp;&nbsp;&nbsp;&nbsp; <b>Positives:</b> {bus_pos_summary} <br/>
    """
content.append(Paragraph(bus_pos_summary_data))

# -------------------------------------------------------------
bus_neg_summary = """This is bussiness' negative Review Summary."""
bus_neg_summary_data = f"""
    &nbsp;&nbsp;&nbsp;&nbsp; <b>Negatives:</b> {bus_neg_summary} <br/>
    """
content.append(Paragraph(bus_neg_summary_data))

# -------------------------------------------------------------
# -------------------------------------------------------------
# # Add some space
# content.append(Spacer(1, 12))

# Add solid horizontal line
content.append(HRFlowable(width="100%", thickness=1, lineCap='round', color='black', 
                            spaceBefore=10, spaceAfter=10))

# -------------------------------------------------------------
# -------------------------------------------------------------
neigh_rev_summary = """Neighbourhood Business' Review Summary: """
neigh_pos_summary_data = f"""
    <b> {neigh_rev_summary} </b><br/>
    """
content.append(Paragraph(neigh_pos_summary_data))

neigh_pos_summary = """This is neigbourhood bussiness' positive review summary."""
neigh_pos_summary_data = f"""
    &nbsp;&nbsp;&nbsp;&nbsp; <b>Positives:</b> {neigh_pos_summary} <br/>
    """
content.append(Paragraph(neigh_pos_summary_data))


neigh_neg_summary = """This is neighbourhod bussiness' negative Review Summary."""
neigh_neg_summary_data = f"""
    &nbsp;&nbsp;&nbsp;&nbsp; <b>Negatives:</b> {neigh_neg_summary} <br/>
    """
content.append(Paragraph(neigh_neg_summary_data))
# -------------------------------------------------------------
# -------------------------------------------------------------
# Add some space
# content.append(Spacer(1, 12))

# Add solid horizontal line
content.append(HRFlowable(width="100%", thickness=1, lineCap='round', color='black', 
                            spaceBefore=10, spaceAfter=10))

# -------------------------------------------------------------
# -------------------------------------------------------------

actionable_insights = """Actionable Insights: """
# actions = """This is the list of actionable insights generated from the user reviews of the business."""
with open('actions.txt', 'r', encoding='utf-8') as f:
    actions = f.read()
    
def parse_text(text):
    points = re.split(r'\d+\.\s', text)
    points = [point.strip() for point in points if point.strip()]
    return points

parsed_actions = parse_text(actions)
for action in parsed_actions:
    title, description = action.split(':')
    title = title.replace('**', '')
    description = description.replace('-', '')
    
    action_formatted = f"""
        <b> {title} </b><br/>
        &nbsp;&nbsp;&nbsp;&nbsp;  {description}
        """
    content.append(Paragraph(action_formatted))


# Add solid horizontal line
content.append(HRFlowable(width="100%", thickness=1, lineCap='round', color='black', 
                            spaceBefore=10, spaceAfter=10))

# -------------------------------------------------------------
# -------------------------------------------------------------
'''
# Generate a chart with matplotlib
fig, ax = plt.subplots()
categories = ['A', 'B', 'C', 'D']
values = [15, 30, 45, 10]
ax.bar(categories, values)
ax.set_title('Sales by Category')

# Save the chart to a bytes buffer
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
# Close the buffer
buf.close()

# Add the chart to the PDF
chart = Image(buf, width=400, height=200)
content.append(chart)
'''
# -------------------------------------------------------------

# Build the PDF
pdf.build(content)

