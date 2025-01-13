import os
import time
import torch
import cv2
import numpy as np
import concurrent.futures

from PIL import Image
from pathlib import Path as SysPath
from typing import List, Optional

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from diffusers import FluxInpaintPipeline

from cog import BasePredictor, Path, Input


# -----------------------------
# Global Constants & Helpers
# -----------------------------

HAND_MODEL_REPO = "Bingsu/adetailer"
HAND_MODEL_FILENAME = "hand_yolov8n.pt"

FLUX_INPAINT_REPO = "black-forest-labs/FLUX.1-dev"

# Optional local caching directory (feel free to customize)
MODEL_CACHE = "checkpoints"

os.makedirs(MODEL_CACHE, exist_ok=True)


def download_with_hf_hub(repo_id: str, filename: str) -> str:
    """Downloads a file from the HF Hub to local disk and returns the local path."""
    start_time = time.time()
    print(f"[~] Downloading `{filename}` from `{repo_id}`...")
    local_path = hf_hub_download(repo_id, filename, cache_dir=MODEL_CACHE)
    print(f"[+] Downloaded `{filename}` in {time.time() - start_time:.2f}s -> {local_path}")
    return local_path


def create_mask_from_yolo(
    input_image: Image.Image,
    yolo_model: YOLO,
    detection_conf_threshold: float = 0.25
) -> Image.Image:
    """
    Runs YOLO hand detection on input_image, returns a single-channel ('L') mask
    with hands in white (255) and the background black (0).
    """
    # Convert PIL to OpenCV
    cv_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    # Run YOLO inference
    results = yolo_model.predict(cv_image, conf=detection_conf_threshold)
    preds = results[0]  # We only have one image in the batch.

    # Prepare a black mask with the same size
    mask = np.zeros((cv_image.shape[0], cv_image.shape[1]), dtype=np.uint8)

    # Each prediction in YOLO v8 has bounding boxes in `preds.boxes.xyxy`
    # shape: (N, 4) -> [xmin, ymin, xmax, ymax]
    if len(preds.boxes) == 0:
        # No hands detected
        return Image.fromarray(mask, mode="L")

    # Fill each bounding box on the mask with white (255)
    for box in preds.boxes.xyxy:
        xmin, ymin, xmax, ymax = box.tolist()
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), color=255, thickness=-1)

    return Image.fromarray(mask, mode="L")


# -----------------------------
# Main Predictor
# -----------------------------

class Predictor(BasePredictor):
    def setup(self):
        """
        Called once at the start of each container.  
        We'll download models concurrently and initialize them.
        """
        print("[~] Setting up models...")

        # Prepare concurrency tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_hand = executor.submit(download_with_hf_hub, HAND_MODEL_REPO, HAND_MODEL_FILENAME)
            future_flux = executor.submit(
                FluxInpaintPipeline.from_pretrained,
                FLUX_INPAINT_REPO,
                torch_dtype=torch.bfloat16,
            )

            # Retrieve YOLO path
            self.hand_model_path = future_hand.result()
            # Retrieve FluxInpaintPipeline
            self.inpaint_pipe = future_flux.result()

        # Initialize YOLO model
        self.hand_detector = YOLO(self.hand_model_path)

        # Move the pipeline to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inpaint_pipe.to(self.device)
        # Offload CPU if you want to minimize VRAM usage
        # self.inpaint_pipe.enable_model_cpu_offload()

        print("[+] Setup complete.")

    def predict(
        self,
        image: Path = Input(description="Input image that contains hands to fix."),
        prompt: str = Input(description="Prompt to guide how the hands should be inpainted.",
                            default="realistic hands, detailed fingers"),
        detection_conf_threshold: float = Input(
            description="Confidence threshold for YOLO hand detection, between 0.0 and 1.0",
            default=0.25,
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale (higher = stricter adherence to prompt)",
            default=7.0,
            ge=1.0,
            le=20.0
        ),
        num_inference_steps: int = Input(
            description="Number of diffusion steps for inpainting",
            default=30,
            ge=1,
            le=100
        ),
        strength: float = Input(
            description="Strength of inpainting for the masked area (0.0 ~ 1.0). "
                        "Higher strength allows for more deviation from the original image in the masked region.",
            default=0.85,
            ge=0,
            le=1
        ),
        height: int = Input(
            description="Height of output image (rounded to the nearest multiple of 8).",
            default=1024,
            ge=128,
            le=2048
        ),
        width: int = Input(
            description="Width of output image (rounded to the nearest multiple of 8).",
            default=1024,
            ge=128,
            le=2048
        ),
        seed: Optional[int] = Input(
            description="Random seed. Leave blank to randomize",
            default=None
        ),
    ) -> Path:
        """
        1) Detect hands in `image` using YOLO.  
        2) Create a hand mask.  
        3) Inpaint them using the FluxInpaintPipeline with `prompt`.  
        4) Return single image path.  
        """
        # Load the input image
        input_image = Image.open(str(image)).convert("RGB")

        # Generate the mask
        mask_image = create_mask_from_yolo(input_image, self.hand_detector, detection_conf_threshold)

        # Round height and width to multiples of 8 (the pipeline often requires that).
        height = (height + 7) // 8 * 8
        width = (width + 7) // 8 * 8

        # If user didn't supply a seed, randomize one.
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"[~] Using seed: {seed}")

        # Construct a torch Generator for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Perform the inpainting
        print("[~] Running inpainting on hand regions...")
        result = self.inpaint_pipe(
            prompt=prompt,
            image=input_image,
            mask_image=mask_image,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=generator,
        ).images[0]

        # Save to a temporary path
        output_path = "/tmp/out_hand_fixed.png"
        result.save(output_path)

        print(f"[+] Done. Saved to {output_path}")
        return Path(output_path)
