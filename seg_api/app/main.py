import uvicorn

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

import numpy as np

import os
import os.path as osp
import uuid
from datetime import datetime
from predictor import seg, save_masked_image

from mobile_sam import sam_model_registry, SamPredictor
from PIL import Image

import time
import random
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile
from fastapi.responses import JSONResponse
import io
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
# 리액트 post 관련 함수
@app.post("/upload") 
async def upload_file(file: UploadFile = File(...)):
    try:
        print("연결완료")
        print(file.filename, "aaaa")
        contents = await file.read()
        img_root = "/project/storage/seg/images"    ### 나중에 상대경로로 수정
        with open(img_root + file.filename, "wb") as f:
            f.write(contents)
        print("file name",  file.filename)
        return {"filename": file.filename}
    except Exception as e:
        print("오류 발생:", e)
        return {"error": str(e)}

def model_define():
    sam_checkpoint = '/project/storage/model/mobile_sam.pt'
    model_type = "vit_t"
    device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor
predictor = model_define()


def from_image_to_bytes(img):
    """
    pillow image 객체를 bytes로 변환
    """
    # Pillow 이미지 객체를 Bytes로 변환
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format=img.format)
    imgByteArr = imgByteArr.getvalue()
    # Base64로 Bytes를 인코딩
    encoded = base64.b64encode(imgByteArr)
    # Base64로 ascii로 디코딩
    decoded = encoded.decode('ascii')
    return decoded


@app.post("/seg")
async def upload_file(x:str, y:str, img:UploadFile = File(...)):
    print("요청 완료")
    x = int(x)
    y = int(y)
    
    print(x,y,type(img))
    input_root = "/project/storage/seg/input" #"./app/input"
    save_root = "/project/storage/seg/seg_db"
    ### byte decode
    file_content = await img.read()  
    im = Image.open(io.BytesIO(file_content))
    ##
    # uuid
    file_uuid = str(uuid.uuid4())
    im_name = file_uuid + '.jpg'
    input_path = os.path.join(input_root, im_name)
    image = im.convert("RGB")
    image.save(input_path, 'JPEG')
    print(111111) 

    image = np.array(image)
    print(image.shape) 
    start = time.time()
    masks, scores, logits = seg(predictor, image, x, y)
    end = time.time()
    print("aaaa", end - start)
    save_paths = []  # 저장된 파일 경로 리스트

    for i in range(3):
        # 파일 이름과 확장자 분리
        data_file_name = f"{file_uuid}_{i+1}.jpg"
        print(i,"번째", data_file_name)
        save_path = os.path.join(save_root, data_file_name) 
        print("save_path:",save_path)
        save_masked_image(image, masks[i], save_path)
        save_paths.append(save_path)  # 저장된 파일 경로 추가
    print(save_path, "완료")
    img_list = []
    for i in range(3):
        img = Image.open(save_paths[i])
        with io.BytesIO() as bf:
            img.save(bf, format='JPEG')
            img_bytes = bf.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        img_list.append(img_base64)
    print("finish")
    return  img_list
    
@app.get("/")
def main():
    return  12

if __name__ == "__main__":
    uvicorn.run("main:app", port=30006, host="0.0.0.0", reload=False)
