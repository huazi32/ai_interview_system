import openpyxl
from fastapi import UploadFile
from io import BytesIO
from service.interview_service import analyze_audio
from api.schemas import InterviewRecord

def import_excel(file: UploadFile):
    wb = openpyxl.load_workbook(BytesIO(file.file.read()))
    ws = wb.active
    for row in ws.iter_rows(min_row=2, values_only=True):
        item = InterviewRecord(
            student_name=row[0], job_name=row[1], round_type=row[2],
            city=row[3], interview_time=row[4], reporter=row[5]
        )
        analyze_audio(item)
    return {"msg": "导入成功"}