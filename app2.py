import os, json, torch
from torch import nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU6(inplace=True)
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.pw(self.dw(x))

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        c = lambda i,o,s: DepthwiseSeparableConv(i,o,s)

        layers = [
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            c(32,  64, 1),
            c(64, 128, 2),
            c(128,128, 1),
            c(128,256, 2),
            c(256,256, 1),
            c(256,512, 2),
        ] + [c(512,512,1) for _ in range(5)] + [
            c(512,1024,2),
            c(1024,1024,1)
        ]

        self.features   = nn.Sequential(*layers)
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(x)

with open('data/pest_info.json', 'r') as f:
    PEST_INFO = json.load(f)

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt       = torch.load("pest_classifier2.pth", map_location=device)

model = MobileNetV1(num_classes=len(ckpt['classes']))
model.load_state_dict(ckpt['model_state_dict'])
model.to(device).eval()

idx_to_class = {v:k for k,v in ckpt['class_to_idx'].items()}

print("✓ MobileNet-v1 loaded")
print(f"✓ Best val-acc: {ckpt['best_acc']*100:.2f}%")
print(f"✓ Classes: {ckpt['classes']}")

transform = transforms.Compose([
    transforms.Resize((128,128)),          # same size your model trained on
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    img  = Image.open(file.stream).convert('RGB')
    x    = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)
        conf, pred = prob.max(1)

    pest_type = idx_to_class[pred.item()]
    response  = {
        "pest"       : pest_type,
        "confidence" : f"{conf.item():.2%}",
        "treatments" : PEST_INFO.get(pest_type, ["Refer to pest_info.json"])
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
