from flask import Flask, request

app = Flask(__name__)

UPLOAD_PATH = '/home/ubuntu/files'

@app.route('/', methods = ['POST'])
def read_plate():

    if 'file' not in request.plate:
        return 'No se ha seleccionado un archivo válido.', 400
    file = request.plate['file']

    if file.filename == ''
        return 'No se ha seleccionado un archivo válido.', 400
    content = file.stream.read()
    return content


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)