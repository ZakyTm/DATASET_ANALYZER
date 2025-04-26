import pandas as pd
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import plotly.io as pio
import mimetypes
class FileHandler:
    def __init__(self):
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
        self.env = Environment(loader=FileSystemLoader('utils/templates'))
        self.loaders = {
            'text/csv': self.load_csv_file,
            'text/plain': self.load_text_file,
            'application/json': self.load_json_file,
            'application/vnd.ms-excel': self.load_csv_file,  # Handle legacy Excel MI types
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self.load_xlsx_file,
            'application/vnd.ms-excel': self.load_xls_file,
            # ... add more file types to the dictionary ...
        }
        mimetypes.add_type('text/csv', '.csv')
        mimetypes.add_type('application/vnd.ms-excel', '.xls')
    
    
    def load_data(self, file_path):
        # First check file extension directly
        if file_path.lower().endswith('.csv'):
            return self.load_csv_file(file_path)
            
        # Fallback to MIME detection
        file_type, _ = mimetypes.guess_type(file_path)
        if file_type in self.loaders:
            return self.loaders[file_type](file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type or 'unknown'}")
    
    def load_csv_file(self, file_path):
        # Add error handling for encoding issues
        try:
            return pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding='latin-1')
    
    def load_text_file(self, file_path):
        # implementation to load text file
        with open(file_path, 'r') as f:
            return f.read()


    def load_json_file(self, file_path):
        # implementation to load json file
        import json
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def load_xlsx_file(self, file_path):
        import pandas as pd
        return pd.read_excel(file_path)
    
    def load_xls_file(self, file_path):
        import pandas as pd
        return pd.read_excel(file_path, engine='xlrd')
    
    def export_pdf(self, report_data):
        try:
            pdf_path = self.report_dir / "full_report.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            
            # Add text content
            c.drawString(100, 750, "Data Analysis Report")
            c.drawString(100, 730, f"Generated at: {pd.Timestamp.now()}")
            
            # Add plots
            y_position = 700
            for plot in report_data.get('plots', []):
                img_path = self.report_dir / plot
                if img_path.exists():
                    c.drawImage(str(img_path), 100, y_position-200, width=400, height=200)
                    y_position -= 220
                    if y_position < 100:
                        c.showPage()
                        y_position = 750
            
            c.save()
            return pdf_path
        except Exception as e:
            raise RuntimeError(f"PDF export failed: {str(e)}")

    def export_html(self, report_data):
        try:
            template = self.env.get_template('report_template.html')
            html_content = template.render(report_data)
            html_path = self.report_dir / "interactive_report.html"
            with open(html_path, 'w') as f:
                f.write(html_content)
            return html_path
        except Exception as e:
            raise RuntimeError(f"HTML export failed: {str(e)}")

    def export_csv(self, data):
        try:
            csv_path = self.report_dir / "processed_data.csv"
            data.to_csv(csv_path, index=False)
            return csv_path
        except Exception as e:
            raise RuntimeError(f"CSV export failed: {str(e)}")
        
    
    def save_report(self, report):
        try:
            report_path = self.report_dir / "data_profile.html"
            report.to_file(report_path)
            return report_path
        except Exception as e:
            raise RuntimeError(f"Report save failed: {str(e)}")
    
    def save_plot(self, fig, name):
        try:
            safe_name = self.get_safe_filename(name)
            html_path = self.report_dir / f"{safe_name}.html"
            png_path = self.report_dir / f"{safe_name}.png"
            
            fig.write_html(html_path)
            fig.write_image(png_path)
            
            return str(png_path)
        except Exception as e:
            raise RuntimeError(f"Plot save failed: {str(e)}")

    @staticmethod
    def get_safe_filename(filename):
        return "".join(c if c.isalnum() else "_" for c in filename)