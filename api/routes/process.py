from fastapi import APIRouter, UploadFile, File
from analysis.statistical_analyzer import statistical_analyzer

router = APIRouter()

@router.post("/process/")
async def process_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = FileHandler.load_from_bytes(content)
    analysis = statistical_analyzer.analyze_dataset(df)
    return analysis
