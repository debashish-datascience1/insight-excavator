from fastapi import FastAPI, UploadFile, File, HTTPException
from app.pipeline.ingest import load_file
from app.pipeline.controller import run_pipeline

app = FastAPI(title="Insight Excavator", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = load_file(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    state = run_pipeline(df, file.filename)
    if state.error:
        raise HTTPException(status_code=500, detail=state.error)
    return state.model_dump()
