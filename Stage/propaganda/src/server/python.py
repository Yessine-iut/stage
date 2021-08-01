# -*- coding: utf-8 -*-
from flask import Flask,request,jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
@app.route("/post",methods=['POST'])
def hello():
    if request.method=='POST':
       request_data = request.get_json()
       print(request_data)
       data = request_data['text']
       print(data)
       return  jsonify(data)
  

if __name__ == "__main__":
    app.run(debug=True,#host=app.config.get("HOST", "127.0.0.2"),
    )