from fastapi import FastAPI , HTTPException,File, UploadFile
import uvicorn
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

device = ('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class Model(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(3, 128 , kernel_size=3 ,padding =1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128*64*64 ,254 ),
        nn.ReLU(),
        nn.Linear(254 , 6),
    )
  def forward(self, x):
    x = self.first(x)
    x =self.second(x)
    return x

classes = torch.load('classes.pt')

model = Model()
model.load_state_dict(torch.load('model_gar.pth', map_location=device))
model.to(device)
model.eval()
garbage_app = FastAPI()

@garbage_app.post('/predict/')
async def check_image(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='файл пустой')
        img_open = Image.open(io.BytesIO(data))
        img_ten = transform(img_open).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_ten)
            pred = y_pred.argmax(dim=1).item()
            return {
                'class': classes[pred]

            }
    except HTTPException as e:
        raise HTTPException(status_code=400, detail=str(e))
if __name__ == '__main__':
    uvicorn.run(garbage_app , port = 8000 , host='127.0.0.1')