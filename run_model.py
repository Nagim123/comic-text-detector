import argparse
import torch
from inference import model2annotations

def detect_text(dir_with_images: str, output_path: str) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("YOU USE CPU DEVICE, you'll wait forever instead of 3 seconds")
    model2annotations(r'data/comictextdetector.pt', dir_with_images, output_path, save_json=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="File to run comic-text-detector!")
    parser.add_argument("--image_dir", type=str, help="Directory with images.")
    parser.add_argument("--output_dir", type=str, help="Directory where to save result of inference.")

    args = parser.parse_args()

    detect_text(args.image_dir, args.output_dir)
