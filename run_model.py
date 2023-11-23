import argparse
import torch
from inference import model2annotations

def detect_text(dir_with_images: str, output_path: str, model_path: str) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model2annotations(model_path, dir_with_images, output_path, save_json=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="File to run comic-text-detector!")
    parser.add_argument("--image_dir", type=str, help="Directory with images.")
    parser.add_argument("--output_dir", type=str, help="Directory where to save result of inference.")
    parser.add_argument("--model_path", type=str, help="Path to .pt model.")

    args = parser.parse_args()

    if args.image_dir is None:
        raise Exception("Image directory is not set! Please provice --image_dir argument")
    if args.output_dir is None:
        raise Exception("Output directory is not set! Please provice --output_dir argument")
    if args.model_path is None:
        raise Exception("Model directory is not set! Please provice --model_path argument")
    
    detect_text(args.image_dir, args.output_dir, args.model_path)
