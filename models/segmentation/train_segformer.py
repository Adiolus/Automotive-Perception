import os
import torch
import pytorch_lightning as pl
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# We will use the 'nvidia/mit-b0' pre-trained weights as our lightweight base model
MODEL_NAME = "nvidia/mit-b0"

class IDDDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        
        self.image_paths = []
        self.mask_paths = []
        
        # 1. Loop through each subfolder
        for folder_name in sorted(os.listdir(image_dir)):
            img_folder = os.path.join(image_dir, folder_name)
            mask_folder = os.path.join(mask_dir, folder_name)
            
            if not os.path.isdir(img_folder):
                continue
                
            # 2. Get all images in this subfolder
            for img_name in sorted(os.listdir(img_folder)):
                if "_leftImg8bit" in img_name:
                    img_path = os.path.join(img_folder, img_name)
                    
                    # 3. Grab the unique ID before "_leftImg8bit"
                    base_name = img_name.split("_leftImg8bit")[0]
                    
                    # 4. Try the most common IDD mask naming conventions
                    possible_masks = [
                        f"{base_name}_gtFine_labelids.png",
                        f"{base_name}_gtFine_labelIds.png",
                        f"{base_name}_gtFine_labellevel3Ids.png"
                    ]
                    
                    # 5. Add to our lists if we find a matching mask
                    for mask_name in possible_masks:
                        mask_path = os.path.join(mask_folder, mask_name)
                        if os.path.exists(mask_path):
                            self.image_paths.append(img_path)
                            self.mask_paths.append(mask_path)
                            break
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # We already have the full, exact paths saved from __init__!
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path) 
        
        # The processor handles the resizing and normalization for SegFormer
        inputs = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")
        
        # Remove the extra batch dimension added by the processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs
    
class SegformerFinetuner(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        # Load the pre-trained SegFormer model architecture 
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME, 
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        images, masks = batch["pixel_values"], batch["labels"]
        outputs = self(pixel_values=images, labels=masks)
        
        # SegFormer outputs logits that are smaller than the original image, 
        # so they must be upsampled to calculate the loss accurately.
        loss, logits = outputs.loss, outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # AdamW is the standard optimizer for Transformer-based models
        return torch.optim.AdamW(self.model.parameters(), lr=2e-5)

if __name__ == "__main__":
    # 1. Load the image processor
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    
    # 2. Point the script to your newly extracted IDD folders
    train_dataset = IDDDataset(
        image_dir="datasets/leftImg8bit/train", 
        mask_dir="datasets/gtFine/train", 
        processor=processor
    )
    
    # 3. Load data in small batches to protect hardware memory
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # 4. Initialize the model for IDD's 41 specific Indian road classes
    model = SegformerFinetuner(num_classes=41) 
    
    # 5. Start the training!
    trainer = pl.Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(model, train_dataloader)