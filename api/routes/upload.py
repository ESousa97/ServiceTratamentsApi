from fastapi import APIRouter, UploadFile, File
from core.file_handler import FileHandler

router = APIRouter()

@router.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = FileHandler.load_from_bytes(content)
    return {"rows": len(df), "columns": len(df.columns), "filename": file.filename}
