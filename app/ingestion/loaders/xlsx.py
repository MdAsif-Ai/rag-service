from typing import List
from openpyxl import load_workbook
from app.ingestion.loaders.base import DocumentLoader, ParsedSection

class XLSXLoader(DocumentLoader):
    def load(self, file_path: str) -> List[ParsedSection]:
        wb = load_workbook(filename=file_path, read_only=True, data_only=True)
        sections = []
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            rows = []
            for row in ws.iter_rows(values_only=True):
                if any(cell is not None for cell in row):
                    row_str = ",".join(str(cell) if cell is not None else "" for cell in row)
                    rows.append(row_str)
            if rows:
                sections.append(ParsedSection(
                    content="\n".join(rows),
                    section=sheet,
                    source_type="xlsx",
                    content_type="table"
                ))
        wb.close()
        return sections