
'''
    Code to generate a PDF report containing the actionable insights generated from the user reviews of the business.
    
    __author__ = ''
    __email__ = ''
    __date__ = ''
    __version__ = ''
'''
import os
import io
import re
import sys
from typing import List, Dict, Any, Optional

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


def build_pdf(config: Dict, bus_id: str, bus_name: str, city: str, category: str, bus_pos_summary: str, 
              bus_neg_summary: str, neigh_pos_summary: str, neigh_neg_summary: str, 
              actions_file_path: str, pdf_save_path: str) -> None:
    '''
        Function to build a PDF report from the information passed such as business information, review summaires, 
            actionable insights, etc.
        
        Args:
            config (Dict): Configuration dictionary
            bus_id (str): Business ID
            bus_name (str): Business Name
            city (str): City
            category (str): Category
            bus_pos_summary (str): Business' positive review summary
            bus_neg_summary (str): Business' negative review summary
            neigh_pos_summary (str): Neighbourhood Business' positive review summary
            neigh_neg_summary (str): Neighbourhood Business' negative review summary
            actions_file_path (str): Path to the text file containing the actionable insights
            pdf_save_path (str): Path to save the PDF report
        Returns:
            None    
    '''
    pdf = SimpleDocTemplate(pdf_save_path, pagesize=A4)

    styles = getSampleStyleSheet()
    styleN = styles["BodyText"]
    styleH = styles["Heading1"]

    content = []

    # -------------------------------------------------------------
    
    # Add the title
    center_aligned_style = ParagraphStyle(
        name='CenterAligned',
        parent=styleH,
        alignment=TA_CENTER
    )
    content.append(Paragraph("User Review Insights", center_aligned_style))

    # -------------------------------------------------------------
    # Add Subtitle - Business ID, Business Name, City, Category
    
    # bus_id = '12345'
    # bus_name = 'Test Business'
    # city = 'Test City'
    # category = 'Test Category'
    subtitle = f"""
            <b>Business ID:</b> {str(bus_id)} <br/>
            <b>Name:</b> {bus_name} <br/>
            <b>City:</b> {city} <br/>
            <b>Categories:</b> {category}
            """
    
    left_aligned_style = ParagraphStyle(
        name='LeftAligned',
        parent=styleN,
        alignment=TA_LEFT
    )
    content.append(Paragraph(subtitle, left_aligned_style))

    # -------------------------------------------------------------
    # Blank space
    content.append(Spacer(1, 12))

    # -------------------------------------------------------------
    # Intro paragraph
    text = """This report contains the actionable insights generated from the user reviews of the business."""
    content.append(Paragraph(text, styleN))

    # -------------------------------------------------------------
    # Add some space
    content.append(Spacer(1, 12))

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # Paragraph with the business' review summary 
    bus_rev_summary = """Business' review summary: """
    bus_pos_summary_data = f"""<b> {bus_rev_summary} </b><br/>"""
    content.append(Paragraph(bus_pos_summary_data))

    # bus_pos_summary = """This is bussiness' positive review summary."""
    bus_pos_summary_data = f"""&nbsp;&nbsp;&nbsp;&nbsp; <b>Positives:</b> {bus_pos_summary} <br/>"""
    content.append(Paragraph(bus_pos_summary_data))

    bus_neg_summary = """This is bussiness' negative Review Summary."""
    bus_neg_summary_data = f"""&nbsp;&nbsp;&nbsp;&nbsp; <b>Negatives:</b> {bus_neg_summary} <br/>"""
    content.append(Paragraph(bus_neg_summary_data))

    # -------------------------------------------------------------
    # horizontal line
    content.append(HRFlowable(width="100%", thickness=1, lineCap='round', color='black', 
                              spaceBefore=10, spaceAfter=10))

    # -------------------------------------------------------------
    # Add a paragraph with the neighbourhood business' review summary
    neigh_rev_summary = """Neighbourhood Business' Review Summary: """
    neigh_pos_summary_data = f"""<b> {neigh_rev_summary} </b><br/>"""
    content.append(Paragraph(neigh_pos_summary_data))

    neigh_pos_summary = """This is neigbourhood bussiness' positive review summary."""
    neigh_pos_summary_data = f"""&nbsp;&nbsp;&nbsp;&nbsp; <b>Positives:</b> {neigh_pos_summary} <br/>"""
    content.append(Paragraph(neigh_pos_summary_data))


    neigh_neg_summary = """This is neighbourhod bussiness' negative Review Summary."""
    neigh_neg_summary_data = f"""&nbsp;&nbsp;&nbsp;&nbsp; <b>Negatives:</b> {neigh_neg_summary} <br/>"""
    content.append(Paragraph(neigh_neg_summary_data))
    
    # -------------------------------------------------------------
    # horizontal line
    content.append(HRFlowable(width="100%", thickness=1, lineCap='round', color='black', 
                                spaceBefore=10, spaceAfter=10))
    # -------------------------------------------------------------
    # Add a paragraph with the actionable insights
    
    # Parse the read text file in 'actions' to get the list of actionable insights
    # Each pointer is separated by a blank line
    '''
    Expected format of the text file:
        Point 1:
            - Actionable Insight 1
            - Actionable Insight 2
    
        Point 2:
            - Actionable Insight 1
            - Actionable Insight 2
    
        etc. 
    '''             
    with open(actions_file_path, 'r', encoding='utf-8') as f:
        actions = f.read()
    
    def parse_text(text):
        points = re.split(r'\n\n', text)
        for point in points:
            if not point.strip():
                points.remove(point)
        return points
    
    parsed_actions = parse_text(actions)
    
    for action in parsed_actions:
        title, description = action.split(':')
        title = title.replace('**', '')
        description = description.replace('-', '')
        
        action_formatted = f"""
            <b> {title} </b><br/>
            &nbsp;&nbsp;&nbsp;&nbsp;  {description}
            <br/><br/>
            """
        content.append(Paragraph(action_formatted))

    # -------------------------------------------------------------
    # Add solid horizontal line
    content.append(HRFlowable(width="100%", thickness=1, lineCap='round', color='black', 
                                spaceBefore=10, spaceAfter=10))
    # -------------------------------------------------------------
    
    # Build the PDF
    pdf.build(content)
    print(f"PDF report saved at: {pdf_save_path}")


def generate_pdf(bus_id: str, bus_name: str, city: str, category: str, bus_pos_summary: str, bus_neg_summary: str, 
                 neigh_pos_summary: str, neigh_neg_summary: str, actions_file_path: str) -> None:
    '''
        Accepts arguments to build a pdf report; gathers required config and makes a call to the build_pdf 
        function to generate the report.
    '''
    config_dict = Config().get_config()
    pdf_save_path = os.path.join(config_dict['output_dir'], 'report.pdf')
    build_pdf(config_dict, bus_id, bus_name, city, category, bus_pos_summary, bus_neg_summary, 
              neigh_pos_summary, neigh_neg_summary, actions_file_path, pdf_save_path)


# if __name__ == '__main__':
#     generate_report()
