# !pip install torch torchvision opencv-python numpy 

import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, ToPILImage

# Load AnimeGAN model from TorchHub
print("Loading AnimeGAN model from TorchHub...")
model = torch.hub.load('bryandlee/animegan2-pytorch', 'generator', pretrained='face_paint_512_v2').eval()
print("Model loaded successfully.")

# Define transformations
transform = Compose([
    Resize((512, 512)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
to_pil = ToPILImage()

# Start webcam
print("Starting webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB and preprocess
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = to_pil(rgb)
    input_tensor = transform(pil_img).unsqueeze(0)

    # Run through AnimeGAN model
    with torch.no_grad():
        out_tensor = model(input_tensor)[0]

    # Post-process output
    out_tensor = (out_tensor.clamp(-1, 1) + 1) / 2  # Convert [-1,1] range to [0,1]
    out_img = to_pil(out_tensor.cpu())
    out_bgr = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)

    # Display the result
    cv2.imshow("AnimeGAN - Live Webcam", out_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("webcam Stopped...")
        break

cap.release()
cv2.destroyAllWindows()
