from fastapi import FastAPI
from Searcher import Searcher


app = FastAPI()
s = Searcher()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/search")
async def search(q: str):
    return s.search(q)
