from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from chat import rag_pipeline  
import shutil
import os

app = FastAPI()

UPLOAD_DIRECTORY = "uploaded_pdfs"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True) 

@app.get("/chatbot_page", response_class=HTMLResponse)
async def chatbot_page(): 
    with open("chatbot.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.post("/uploadfile/", response_class=JSONResponse)
async def create_upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        
        return {"info": f"File '{file.filename}' saved at '{file_location}' on the server.", "filename": file.filename}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Could not upload file: {e}"})
    finally:
        file.file.close() 

@app.post("/message", response_class=JSONResponse)
async def message(file_path: str = Form(...), user_message: str = Form(...)):
    server_side_file_path = os.path.join(UPLOAD_DIRECTORY, file_path)

    if not os.path.exists(server_side_file_path):
        return JSONResponse(
            status_code=404,
            content={"message": f"File '{file_path}' not found on server. Please upload it first."}
        )
    bot_message = rag_pipeline(server_side_file_path, user_message)
    content = {"message": bot_message}
    return JSONResponse(content=content)