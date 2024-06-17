import config
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

#CORSの設定
origins = [
    "http://localhost",
    "http://localhost:8080",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"root": "be-pss"}

if __name__ == "__main__":
    uvicorn.run("server:app",host="0.0.0.0",port=int(config.BACKEND_PORT),reload=True)