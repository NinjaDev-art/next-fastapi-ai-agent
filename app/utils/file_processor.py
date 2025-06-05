import io
import logging
import requests
from pathlib import Path
import docx
from PyPDF2 import PdfReader
import pandas as pd
import json
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import xlrd
import openpyxl
from typing import List, Dict, Any
from ..config.settings import settings

logger = logging.getLogger(__name__)

class FileProcessor:
    def __init__(self):
        self.processors = {
            '.pdf': self.process_pdf,
            '.docx': self.process_docx,
            '.csv': self.process_csv,
            '.txt': self.process_txt,
            '.json': self.process_json,
            '.html': self.process_html,
            '.xls': self.process_xls,
            '.xlsx': self.process_xlsx,
            '.xml': self.process_xml
        }

    def download_file(self, url: str) -> bytes:
        logger.info(f"Downloading file from URL: {url}")
        try:
            if settings.AWS_CDN_URL:
                # Download from AWS CDN
                response = requests.get(f"{settings.AWS_CDN_URL}/{url}")
            else:
                # Download from original URL
                response = requests.get(url)
            response.raise_for_status()
            logger.info(f"Successfully downloaded file from {url}")
            return response.content
        except Exception as e:
            logger.error(f"Error downloading file from {url}: {str(e)}")
            raise

    def process_pdf(self, content: bytes) -> str:
        try:
            pdf_reader = PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += str(page_text)
            return str(text) if text else "No text content found in PDF"
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return f"Error processing PDF file: {str(e)}"

    def process_docx(self, content: bytes) -> str:
        try:
            doc = docx.Document(io.BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text:
                    text += str(paragraph.text) + "\n"
            return str(text) if text else "No text content found in DOCX"
        except Exception as e:
            logger.error(f"Error processing DOCX: {str(e)}")
            return f"Error processing DOCX file: {str(e)}"

    def process_csv(self, content: bytes) -> str:
        try:
            df = pd.read_csv(io.BytesIO(content))
            return str(df.to_string())
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            return f"Error processing CSV file: {str(e)}"

    def process_txt(self, content: bytes) -> str:
        try:
            return str(content.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error processing TXT: {str(e)}")
            return f"Error processing TXT file: {str(e)}"

    def process_json(self, content: bytes) -> str:
        try:
            data = json.loads(content)
            return str(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error processing JSON: {str(e)}")
            return f"Error processing JSON file: {str(e)}"

    def process_html(self, content: bytes) -> str:
        try:
            soup = BeautifulSoup(content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            return str(soup.get_text())
        except Exception as e:
            logger.error(f"Error processing HTML: {str(e)}")
            return f"Error processing HTML file: {str(e)}"

    def process_xls(self, content: bytes) -> str:
        try:
            workbook = xlrd.open_workbook(file_contents=content)
            text = ""
            for sheet in workbook.sheets():
                text += f"Sheet: {sheet.name}\n"
                for row in range(sheet.nrows):
                    row_data = "\t".join(str(cell.value) for cell in sheet.row(row))
                    text += str(row_data) + "\n"
            return str(text)
        except Exception as e:
            logger.error(f"Error processing XLS: {str(e)}")
            return f"Error processing XLS file: {str(e)}"

    def process_xlsx(self, content: bytes) -> str:
        try:
            workbook = openpyxl.load_workbook(io.BytesIO(content))
            text = ""
            for sheet in workbook:
                text += f"Sheet: {sheet.title}\n"
                for row in sheet.iter_rows():
                    row_data = "\t".join(str(cell.value) for cell in row)
                    text += str(row_data) + "\n"
            return str(text)
        except Exception as e:
            logger.error(f"Error processing XLSX: {str(e)}")
            return f"Error processing XLSX file: {str(e)}"

    def process_xml(self, content: bytes) -> str:
        try:
            tree = ET.ElementTree(ET.fromstring(content))
            root = tree.getroot()
            return str(ET.tostring(root, encoding='unicode', method='text'))
        except Exception as e:
            logger.error(f"Error processing XML: {str(e)}")
            return f"Error processing XML file: {str(e)}"

    def process_files(self, files: List[str]) -> str:
        logger.info(f"Processing files: {files}")
        all_text = ""
        for file_url in files:
            try:
                content = self.download_file(file_url)
                file_extension = Path(file_url).suffix.lower()
                logger.info(f"Processing file with extension: {file_extension}")
                
                processor = self.processors.get(file_extension, self.process_txt)
                text = processor(content)
                
                # Ensure text is a string
                if not isinstance(text, str):
                    logger.warning(f"Processor returned {type(text)} instead of string, converting")
                    text = str(text)
                
                all_text += text + "\n\n"
                logger.info(f"Successfully processed file: {file_url}")
            except Exception as e:
                logger.error(f"Error processing file {file_url}: {str(e)}")
                # Continue processing other files instead of raising
                all_text += f"Error processing {file_url}: {str(e)}\n\n"
                
        return all_text
    
    def identify_files(self, files: List[str]) -> List[str]:
        """
        Identify the type of files and return the image files and the text files.
        """
        image_files = []
        text_files = []
        for file_url in files:
            file_extension = Path(file_url).suffix.lower()
            if file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.ico', '.webp']:
                image_files.append(file_url)
            else:
                text_files.append(file_url)
        return image_files, text_files

file_processor = FileProcessor() 