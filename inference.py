import model
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
import albumentations as A


device="cuda" if torch.cuda.is_available() else "cpu"

image="588.jpg"

t_test = A.Resize(768, 1152, interpolation=cv2.INTER_NEAREST)


def predict_image_mask(model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    img = cv2.imread(image)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aug=t_test(image=img)
    img=Image.fromarray(aug['image'])

    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    image = t(img)
    model.to(device); image=image.to(device)

    with torch.no_grad():
        
        image = image.unsqueeze(0)
        
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked

pred_mask = predict_image_mask(model.model, image)

fig= plt.plot(figsize=(20,10))
plt.imshow(pred_mask)
plt.show()