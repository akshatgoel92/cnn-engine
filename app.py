import os
from math import floor
import torchvision.transforms as transforms

from flask import Flask, render_template, request
from train.utils import testing_loop
from train.alexnet import model



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('data', 'test')

CLASS_NAMES={0: 'Fish', 1: 'Dog'}
NUM_CLASSES=len(CLASS_NAMES.keys())
MODEL_PATH=os.path.join('models', 'alexnet_best_loss.pkl')

IMAGE_DIM=227
STDS=[0.229, 0.224, 0.225]
MEANS=[0.485, 0.456, 0.406]

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/infer', methods=['POST'])
def success():

    TRANSFORMS_LIST = [transforms.CenterCrop(IMAGE_DIM),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=MEANS, std=STDS),]
    
    if request.method == 'POST':
        f = request.files['file']
        IMG_PATH = f.filename
        f.save(IMG_PATH)
        prediction = testing_loop(IMG_PATH, NUM_CLASSES, MODEL_PATH, TRANSFORMS_LIST)
        os.remove(IMG_PATH)
        return render_template('inference.html', name=CLASS_NAMES[int(prediction[-1])])


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)