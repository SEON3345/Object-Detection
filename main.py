import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser(); p.add_argument('image'); p.add_argument('--confidence', type=float, default=0.5); args = p.parse_args()
    model = YOLO('yolo11n.pt')
    results = model.predict(args.image, conf=args.confidence, save=True)
    for result in results:
        for box in result.boxes:
            print(result.names[int(box.cls.item())], round(box.conf.item(), 3))
        print('Saved to:', result.save_dir)

if __name__ == '__main__':
    main()
