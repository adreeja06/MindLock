from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from ocr import *
import os

model=load_model("mindlock-model.h5")

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit",methods=['POST'])
def submit():
    l=[]
    for i in range(1,10):
        for j in range(1,10):
            k=request.form[f"cell_{i}{j}"]
            if k == "":
                l.append(0)
            else:
                l.append(int(k))
    isempty=1
    for i in l:
        if i!=0:
            isempty=0
            break
    if isempty==1:
        return render_template("index.html",message="All cells cannot be empty!")
    else:
        print(l)
        lnp=np.array(l).reshape(9,9)
        ln=list(model.predict(lnp.reshape(1,9,9,1)).argmax(-1).squeeze()+1)
        return render_template("solution.html", ans=ln)


@app.route("/uploadview")
def uploadview():
    return render_template('upload.html')

@app.route("/upload",methods=["POST"])
def upload():
    uploaded_file=request.files["image"]
    if uploaded_file.filename!="" and uploaded_file:
        uploaded_file.save("sudoku_image.jpg")
        classes=np.arange(0,10)
        model_new=load_model("model-OCR.h5")
        input_size=48
        board,location= find_board(cv2.imread("sudoku_image.jpg"))
        gray=cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        rois=split_boxes(gray)
        rois=np.array(rois).reshape(-1,input_size,input_size,1)
        prediction=model_new.predict(rois)
        predicted_numbers=[]
        for i in prediction:
            index=np.argmax(i)
            predicted_number=classes[index]
            predicted_numbers.append(predicted_number)
        board_num=np.array(predicted_numbers).astype('uint8').reshape(9,9).tolist()
        for k in range(9):
            for j in range(9):
                if board_num[k][j]==0:
                    board_num[k][j]=''
                else:
                    board_num[k][j]=str(board_num[k][j])
        file_path='sudoku_image.jpg'
        if os.path.exists(file_path):
            os.remove(file_path)
        return render_template('index.html',board=board_num)
    return render_template('upload.html',message="No file selected")