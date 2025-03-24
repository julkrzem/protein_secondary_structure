from flask import Flask, render_template, request, Response
import json
from lstm import *


app = Flask(__name__)

with open('static/input_char2index.json') as f:
    input_lang = json.load(f)
with open('static/output_index2char.json') as f:
    output_lang3 = json.load(f)
with open('static/output_index2char8.json') as f:
    output_lang8 = json.load(f)

color_map = {"C": 'lightsteelblue',
            "E": "royalblue",
            "H": "darkturquoise",
            "B": "seagreen",
            "G": "yellowgreen",
            "I": "tomato",
            "T": "mediumpurple",
            "S": "palegreen"}

model3 = LSTM(22, 64, 256, 4)
model8 = LSTM(22, 64, 256, 9)
model3.load_state_dict(torch.load("Models/lstm_1.pth"))
model8.load_state_dict(torch.load("Models/lstm_8.pth"))
model3.eval()
model8.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        sequence = request.form["content"]
        model_select = request.form["model"]
        if model_select=="sst3":
            model = model3
            output_lang = output_lang3
        else: 
            model = model8
            output_lang = output_lang8

        prediction = model_predict([sequence], input_lang,output_lang,model)

        print_results(color_map, sequence.upper(), prediction[0])

        return render_template('index.html', prediction=prediction[0], sequence=sequence.upper().replace("*","x"))
    else:
        return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        file = request.files['file']
        model_select = request.form["model"]
        if model_select=="sst3":
            model = model3
            output_lang = output_lang3
        else: 
            model = model8
            output_lang = output_lang8
        ids, seq = read_fasta(file)
        prediction = model_predict(seq, input_lang,output_lang,model)
        prev = list(zip(ids, seq, prediction))
 
        return render_template('upload.html', prev=prev)
    
    return render_template('upload.html')

@app.route('/plot')
def plot():
    sequence = request.args.get('sequence')
    prediction = request.args.get('prediction')
    img = print_results(color_map, sequence, prediction)
    return Response(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)


