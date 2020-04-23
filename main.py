import io

import torch
from PIL import Image
from flask import Flask, request
from flask import render_template
from torchvision import transforms
from werkzeug import run_simple

from src.demo.info import class_mapping
from src.demo.nts_net.model import attention_net
from src.read_from_targz import torch_load_targz

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', show_result=False)


@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.files['img'].read()
    image = Image.open(io.BytesIO(image_data))
    pred_id, text_pred = model_predict(image)
    return render_template('index.html', show_result=True, pred_id=pred_id, text_pred=text_pred)


def model_predict(img):
    transform_test = transforms.Compose([
        transforms.Resize((448, 448), Image.BILINEAR),
        transforms.ToTensor(),
    ])
    scaled_img = transform_test(img)
    torch_images = scaled_img.unsqueeze(0)

    net = attention_net(topN=6, num_classes=131, device=(torch.device('cpu')))
    net.load_state_dict(torch_load_targz('models/nts_net_state.tar.gz'))
    net.eval()
    with torch.no_grad():
        _, _, row_logits, concat_logits, _, _, _ = net(torch_images)
        _, concat_predict = torch.max(concat_logits, 1)
        pred_id = concat_predict.item()
        text_pred = class_mapping[pred_id]
        print(f'class id: {pred_id}, class name: {text_pred}')
    return pred_id, text_pred


if __name__ == "__main__":
    run_simple('localhost', 5001, app, use_reloader=True, use_debugger=True)
