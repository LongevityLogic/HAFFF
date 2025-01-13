import os
import time
import torch
import cv2
import numpy as np
import concurrent.futures

from PIL import Image
from pathlib import Path as SysPath
from typing import Optional

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from diffusers import FluxInpaintPipeline

from cog import BasePredictor, Input, Path


# -----------------------------
# Global Constants & Helpers
# -----------------------------

# For example, both YOLO models stored in "Bingsu/adetailer" on HF:
HAND_MODEL_REPO = "Bingsu/adetailer"
HAND_MODEL_FILENAME = "hand_yolov9c.pt"

FACE_MODEL_REPO = "Bingsu/adetailer"
FACE_MODEL_FILENAME = "face_yolov9c.pt"

# FLUX Inpaint pipeline
FLUX_INPAINT_REPO = "black-forest-labs/FLUX.1-dev"

# Local caching directory
MODEL_CACHE = "checkpoints"
os.makedirs(MODEL_CACHE, exist_ok=True)


def download_with_hf_hub(repo_id: str, filename: str) -> str:
    """
    Downloads a file from the HF Hub to local disk and returns the local path.
    """
    start_time = time.time()
    print(f"[~] Downloading `{filename}` from `{repo_id}`...")
    local_path = hf_hub_download(repo_id, filename, cache_dir=MODEL_CACHE)
    print(f"[+] Downloaded `{filename}` in {time.time() - start_time:.2f}s -> {local_path}")
    return local_path


def create_mask_from_yolo(cv_image: np.ndarray, model: YOLO, conf: float) -> Image.Image:
    """
    Runs YOLO detection on an OpenCV image (BGR).
    Creates a single-channel (L) mask for the detected regions (white = 255, black = 0).
    """
    results = model.predict(cv_image, conf=conf)
    preds = results[0]  # YOLO returns a Results object per image

    mask = np.zeros((cv_image.shape[0], cv_image.shape[1]), dtype=np.uint8)

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
        Called once at container start. Downloads YOLO for hands, YOLO for faces,
        and the FLUX inpainting pipeline concurrently, then initializes them.
        """
        print("[~] Setting up models...")

        # Prepare concurrent tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_hand = executor.submit(
                download_with_hf_hub, HAND_MODEL_REPO, HAND_MODEL_FILENAME
            )
            future_face = executor.submit(
                download_with_hf_hub, FACE_MODEL_REPO, FACE_MODEL_FILENAME
            )
            future_flux = executor.submit(
                FluxInpaintPipeline.from_pretrained,
                FLUX_INPAINT_REPO,
                torch_dtype=torch.bfloat16
            )

            # Fetch results
            self.hand_model_path = future_hand.result()
            self.face_model_path = future_face.result()
            self.inpaint_pipe = future_flux.result()

        # Initialize YOLO detection models
        self.hand_detector = YOLO(self.hand_model_path)
        self.face_detector = YOLO(self.face_model_path)

        # Move FLUX pipeline to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inpaint_pipe.to(self.device)

        print("[+] Setup complete.")

    def predict(
        self,
        image: Path = Input(description="Input image to fix hands and faces."),
        prompt_for_hands: str = Input(
            description="Prompt describing how the fixed hands/faces should appear.",
            default="realistic hands, detailed fingers",
        ),
        prompt_for_faces: str = Input(
            description="Prompt describing how the fixed faces should appear.",
            default="realistic faces, sharp features",
        ),
        detection_conf_threshold: float = Input(
            description="Confidence threshold for YOLO detection (0.0 ~ 1.0)",
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
            default=20,
            ge=1,
            le=100
        ),
        strength: float = Input(
            description="Strength of inpainting in masked areas (0.0 ~ 1.0). "
                        "Higher = more deviation from original image.",
            default=0.5,
            ge=0,
            le=1
        ),
        seed: Optional[int] = Input(
            description="Random seed. Leave blank to randomize.",
            default=None
        ),
    ) -> Path:
        """
        1) Load the input image.
        2) Detect hands, inpaint them.
        3) Detect faces (on the hand-fixed image), inpaint them.
        4) Return the fully inpainted image with original dimensions.
        """
        # Load and convert to RGB
        original_image = Image.open(str(image)).convert("RGB")
        orig_width, orig_height = original_image.size

        # Randomize seed if none provided
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"[~] Using seed: {seed}")
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # -------------------------------------------------------------
        # 1) HAND PASS
        # -------------------------------------------------------------
        print("[~] Detecting and inpainting hands...")
        # Convert to OpenCV
        cv_image_hand = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        # Create hand mask
        hand_mask = create_mask_from_yolo(cv_image_hand, self.hand_detector, detection_conf_threshold)
        # Inpaint
        hand_inpainted_image = self.inpaint_pipe(
            prompt=prompt_for_hands,
            image=original_image,         # original image
            mask_image=hand_mask,         # mask for hands
            height=orig_height,           # preserve original size
            width=orig_width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=generator,
        ).images[0]

        # -------------------------------------------------------------
        # 2) FACE PASS
        # -------------------------------------------------------------
        print("[~] Detecting and inpainting faces...")
        # Convert new image (with hands fixed) to OpenCV
        cv_image_face = cv2.cvtColor(np.array(hand_inpainted_image), cv2.COLOR_RGB2BGR)
        # Create face mask
        face_mask = create_mask_from_yolo(cv_image_face, self.face_detector, detection_conf_threshold)
        # Inpaint again
        final_result = self.inpaint_pipe(
            prompt=prompt_for_faces,
            image=hand_inpainted_image,  # the hand-fixed image
            mask_image=face_mask,        # mask for faces
            height=orig_height,
            width=orig_width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=generator,
        ).images[0]

        # -------------------------------------------------------------
        # Save and return
        # -------------------------------------------------------------
        output_path = "/tmp/out_fixed_hands_faces.png"
        final_result.save(output_path)
        print(f"[+] Done. Final image saved to {output_path}")

        return Path(output_path)
