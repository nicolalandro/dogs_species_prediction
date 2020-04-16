import io

import torch
from PIL import Image
from flask import Flask, request
from flask import render_template
from torchvision import transforms
from werkzeug import run_simple

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
    class_mapping = {0: 'affenpinscher', 1: 'afghan_hound', 2: 'african_hunting_dog', 3: 'airedale',
                     4: 'american_bulldog', 5: 'american_pit_bull_terrier', 6: 'american_staffordshire_terrier',
                     7: 'appenzeller', 8: 'australian_terrier', 9: 'basenji', 10: 'basset', 11: 'basset_hound',
                     12: 'beagle', 13: 'bedlington_terrier', 14: 'bernese_mountain_dog', 15: 'blenheim_spaniel',
                     16: 'bloodhound', 17: 'bluetick', 18: 'border_collie', 19: 'border_terrier', 20: 'borzoi',
                     21: 'boston_bull', 22: 'bouvier_des_flandres', 23: 'boxer', 24: 'brabancon_griffon', 25: 'briard',
                     26: 'brittany_spaniel', 27: 'bull_mastiff', 28: 'cairn', 29: 'cardigan',
                     30: 'chesapeake_bay_retriever', 31: 'chihuahua', 32: 'chow', 33: 'clumber', 34: 'coated_retriever',
                     35: 'coated_wheaten_terrier', 36: 'cocker_spaniel', 37: 'collie', 38: 'dandie_dinmont',
                     39: 'dhole', 40: 'dingo', 41: 'doberman', 42: 'english_cocker_spaniel', 43: 'english_foxhound',
                     44: 'english_setter', 45: 'english_springer', 46: 'entlebucher', 47: 'eskimo_dog',
                     48: 'french_bulldog', 49: 'german_shepherd', 50: 'german_shorthaired', 51: 'giant_schnauzer',
                     52: 'golden_retriever', 53: 'gordon_setter', 54: 'great_dane', 55: 'great_pyrenees',
                     56: 'greater_swiss_mountain_dog', 57: 'groenendael', 58: 'haired_fox_terrier',
                     59: 'haired_pointer', 60: 'havanese', 61: 'ibizan_hound', 62: 'irish_setter', 63: 'irish_terrier',
                     64: 'irish_water_spaniel', 65: 'irish_wolfhound', 66: 'italian_greyhound', 67: 'japanese_chin',
                     68: 'japanese_spaniel', 69: 'keeshond', 70: 'kelpie', 71: 'kerry_blue_terrier', 72: 'komondor',
                     73: 'kuvasz', 74: 'labrador_retriever', 75: 'lakeland_terrier', 76: 'leonberg', 77: 'leonberger',
                     78: 'lhasa', 79: 'malamute', 80: 'malinois', 81: 'maltese_dog', 82: 'mexican_hairless',
                     83: 'miniature_pinscher', 84: 'miniature_poodle', 85: 'miniature_schnauzer', 86: 'newfoundland',
                     87: 'norfolk_terrier', 88: 'norwegian_elkhound', 89: 'norwich_terrier', 90: 'old_english_sheepdog',
                     91: 'otterhound', 92: 'papillon', 93: 'pekinese', 94: 'pembroke', 95: 'pomeranian', 96: 'pug',
                     97: 'redbone', 98: 'rhodesian_ridgeback', 99: 'rottweiler', 100: 'saint_bernard', 101: 'saluki',
                     102: 'samoyed', 103: 'schipperke', 104: 'scotch_terrier', 105: 'scottish_deerhound',
                     106: 'scottish_terrier', 107: 'sealyham_terrier', 108: 'shetland_sheepdog', 109: 'shiba_inu',
                     110: 'siberian_husky', 111: 'silky_terrier', 112: 'staffordshire_bull_terrier',
                     113: 'staffordshire_bullterrier', 114: 'standard_poodle', 115: 'standard_schnauzer',
                     116: 'sussex_spaniel', 117: 'tan_coonhound', 118: 'tibetan_mastiff', 119: 'tibetan_terrier',
                     120: 'toy_poodle', 121: 'toy_terrier', 122: 'tzu', 123: 'vizsla', 124: 'walker_hound',
                     125: 'weimaraner', 126: 'welsh_springer_spaniel', 127: 'west_highland_white_terrier',
                     128: 'wheaten_terrier', 129: 'whippet', 130: 'yorkshire_terrier'}
    transform_test = transforms.Compose([
        transforms.Resize((448, 448), Image.BILINEAR),
        transforms.ToTensor(),
    ])
    scaled_img = transform_test(img)
    torch_images = scaled_img.unsqueeze(0)

    net = torch.load('models/nts_net.pt')
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
