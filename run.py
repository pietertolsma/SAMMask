from segment_anything import (
    build_sam,
    SamPredictor,
    sam_model_registry,
    SamAutomaticMaskGenerator,
)
import glob
from tqdm import tqdm

from sammask.mask import predict, predict_2


jpg_files = glob.glob("input_data/**/*.jpg", recursive=True)

# generate_points(1920, 1080)

print("Loading SAM..")
sam = build_sam(checkpoint="checkpoint_large.pth")
predictor = SamPredictor(sam)
# predictor = SamPredictor(sam_model_registry["vit_b"](checkpoint="checkpoint.pth"))

print("Starting process..")
for i in tqdm(range(len(jpg_files))):
    predict(predictor, jpg_files[i], draw_points=False)
    # predict_2(generator, jpg_files[2])
