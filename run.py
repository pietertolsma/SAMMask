from segment_anything import build_sam, SamPredictor
import glob
from tqdm import tqdm

from sammask.mask import predict


jpg_files = glob.glob("input_data/**/*.jpg", recursive=True)

# generate_points(1920, 1080)

print("Loading SAM..")
predictor = SamPredictor(build_sam(checkpoint="checkpoint.pth"))

print("Starting process..")
for i in tqdm(range(len(jpg_files))):
    predict(predictor, jpg_files[i])
