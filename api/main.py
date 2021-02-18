from flask import Flask
from detectron2.structures import BoxMode
app = Flask(__name__)

@app.route('/api/')
def hello_world():
    return 'Hello, World!'