import cv2
import torch
import numpy as np
from torchvision import transforms

# --- THE ONE LINE YOU MIGHT NEED TO CHANGE ---
# You need to import the specific Lightning class you used to train the model.
# Open 'train_segformer.py' and look for the class name (e.g., class SegFormerLightning(pl.LightningModule):)
# Replace 'YourModelClass' below with that exact name.
from train_segformer import SegformerFinetuner 

# 1. Setup paths
ckpt_path = "lightning_logs/version_2/checkpoints/epoch=9-step=34970.ckpt"
input_video = r"C:\Users\balaj\Downloads\huh.mp4" 
output_video = "indian_segmented_output.mp4"

print("Waking up the AI...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load the brain
model = SegformerFinetuner.load_from_checkpoint(ckpt_path, num_classes=41)
model.to(device)
model.eval()

# 3. Setup Video Processing
cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Setup the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print(f"Processing video: {width}x{height} at {fps} FPS")

# Define a color map for the AI to paint with
np.random.seed(42) # Keeps the colors consistent each time you run it
COLORS = np.random.randint(0, 255, size=(41, 3), dtype=np.uint8)

# Basic transform to get the image ready for the model
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)) # Adjust if your model expects a different size
])

frame_count = 0
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}...")

        # Prepare image for AI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame).unsqueeze(0).to(device)

        # Predict
        outputs = model(input_tensor)
        
        # Extract the predictions (handles different SegFormer output formats)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0] 
        
        # Resize AI output back to the original video size
        logits = torch.nn.functional.interpolate(logits, size=(height, width), mode='bilinear', align_corners=False)
        
        # Get the highest probability class for each pixel
        preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        # Paint the pixels!
        color_mask = COLORS[preds % len(COLORS)] 

        # Blend original frame with the painted mask (50% transparency)
        blended = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)
        
        out.write(blended)

cap.release()
out.release()
print(f"Done! Check the folder for {output_video}")
