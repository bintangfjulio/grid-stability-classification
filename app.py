import torch
import pandas as pd
import torch.nn as nn

from flask import Flask, render_template, request, redirect
from model.bilstm import BiLSTM
from util.preprocessor import Preprocessor

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        data = {
            'tau1': [request.form.get('tau1')],
            'tau2': [request.form.get('tau2')],
            'tau3': [request.form.get('tau3')],
            'tau4': [request.form.get('tau4')],
            'p1': [request.form.get('p1')],
            'p2': [request.form.get('p2')],
            'p3': [request.form.get('p3')],
            'p4': [request.form.get('p4')],
            'g1': [request.form.get('g1')],
            'g2': [request.form.get('g2')],
            'g3': [request.form.get('g3')],
            'g4': [request.form.get('g4')]
        }

        df = pd.DataFrame(data).astype('float')

        module = Preprocessor(batch_size=64)
        num_classes, input_size = module.get_feature_size()

        model = BiLSTM.load_from_checkpoint('checkpoints/bilstm_result/epoch=34-step=5040.ckpt', lr=1e-3, num_classes=num_classes, input_size=input_size)
        model.eval()
        model.freeze()
        X = torch.tensor(df.values.tolist())

        with torch.no_grad():
            preds = model(X)

        preds = nn.Sigmoid()(preds.squeeze(1))
        preds = preds.numpy()
        
        if preds > 0.5:
            print('Stable')
        else:
            print('unstable')

        return redirect('/#classifier')

    return render_template('index.html')
    

if __name__ == '__main__':
    app.run(debug=True)