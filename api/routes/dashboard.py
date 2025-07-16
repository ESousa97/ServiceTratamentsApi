from fastapi import APIRouter, UploadFile, File
from visualization.dashboard_generator import DashboardGenerator
from core.file_handler import FileHandler

router = APIRouter()

@router.post("/dashboard/")
async def dashboard_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = FileHandler.load_from_bytes(content)
    dashboard = DashboardGenerator.generate_dashboard(df)
    return dashboard
