import torch
import pandas as pd

from flask import Flask, render_template, request, redirect
from model.bilstm import BiLSTM
from scipy.stats import zscore

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        data = {
            'tau1': [float(request.form.get('tau1'))],
            'tau2': [float(request.form.get('tau2'))],
            'tau3': [float(request.form.get('tau3'))],
            'tau4': [float(request.form.get('tau4'))],
            'p1': [float(request.form.get('p1'))],
            'p2': [float(request.form.get('p2'))],
            'p3': [float(request.form.get('p3'))],
            'p4': [float(request.form.get('p4'))],
            'g1': [float(request.form.get('g1'))],
            'g2': [float(request.form.get('g2'))],
            'g3': [float(request.form.get('g3'))],
            'g4': [float(request.form.get('g4'))]
        }

        X = torch.tensor(pd.DataFrame(data).values.tolist())
        classify(X)

        return redirect('/#classifier')

    return render_template('index.html')

def classify(X):
    model = BiLSTM.load_from_checkpoint('checkpoints/bilstm_result/epoch=34-step=5040.ckpt')

    model.eval()
    X_tensor = torch.tensor(X)

    with torch.no_grad():
        predictions = model(X_tensor)

    predictions = predictions.numpy()
    print(predictions)


if __name__ == '__main__':
    app.run(debug=True)