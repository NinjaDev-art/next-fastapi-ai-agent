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
        pdf_reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def process_docx(self, content: bytes) -> str:
        doc = docx.Document(io.BytesIO(content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def process_csv(self, content: bytes) -> str:
        df = pd.read_csv(io.BytesIO(content))
        return df.to_string()

    def process_txt(self, content: bytes) -> str:
        return content.decode('utf-8')

    def process_json(self, content: bytes) -> str:
        data = json.loads(content)
        return json.dumps(data, indent=2)

    def process_html(self, content: bytes) -> str:
        soup = BeautifulSoup(content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text()

    def process_xls(self, content: bytes) -> str:
        workbook = xlrd.open_workbook(file_contents=content)
        text = ""
        for sheet in workbook.sheets():
            text += f"Sheet: {sheet.name}\n"
            for row in range(sheet.nrows):
                text += "\t".join(str(cell.value) for cell in sheet.row(row)) + "\n"
        return text

    def process_xlsx(self, content: bytes) -> str:
        workbook = openpyxl.load_workbook(io.BytesIO(content))
        text = ""
        for sheet in workbook:
            text += f"Sheet: {sheet.title}\n"
            for row in sheet.iter_rows():
                text += "\t".join(str(cell.value) for cell in row) + "\n"
        return text

    def process_xml(self, content: bytes) -> str:
        tree = ET.ElementTree(ET.fromstring(content))
        root = tree.getroot()
        return ET.tostring(root, encoding='unicode', method='text')

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
                
                all_text += text + "\n\n"
                logger.info(f"Successfully processed file: {file_url}")
            except Exception as e:
                logger.error(f"Error processing file {file_url}: {str(e)}")
                raise
        return all_text

file_processor = FileProcessor() 