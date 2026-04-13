from pathlib import Path
from pypdf import PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

# Create data directory
data_dir = Path('rag_app/data')
data_dir.mkdir(parents=True, exist_ok=True)

# Create a simple PDF
buffer = BytesIO()
c = canvas.Canvas(buffer, pagesize=letter)
c.drawString(100, 750, 'This is a sample PDF for the knowledge assistant.')
c.drawString(100, 730, 'It contains some text to test the ingestion pipeline.')
c.save()
buffer.seek(0)

# Save to file
pdf_path = data_dir / 'sample.pdf'
with open(pdf_path, 'wb') as f:
    f.write(buffer.getvalue())

print('Created sample PDF at', pdf_path)