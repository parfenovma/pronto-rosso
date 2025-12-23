from PIL import Image
import glob
import os
import sys
import argparse


def create_gif_from_args():
    parser = argparse.ArgumentParser(description='Создание GIF из PNG файлов')
    parser.add_argument('input_folder', help='Папка с PNG файлами')
    parser.add_argument('output_file', help='Имя выходного GIF файла')
    parser.add_argument('--duration', type=int, default=200, 
                       help='Длительность кадра в мс (по умолчанию: 200)')
    parser.add_argument('--loop', type=int, default=0,
                       help='Количество повторов (0=бесконечно, по умолчанию: 0)')
    
    args = parser.parse_args()
    
    png_files = sorted(glob.glob(os.path.join(args.input_folder, "*.png")))
    
    if not png_files:
        print(f"Ошибка: В папке '{args.input_folder}' не найдено PNG файлов")
        sys.exit(1)
    
    images = []
    for file in png_files:
        img = Image.open(file)
        images.append(img)
    
    images[0].save(
        args.output_file,
        save_all=True,
        append_images=images[1:],
        duration=args.duration,
        loop=args.loop
    )
    
    for img in images:
        img.close()
    
    print(f"Создан GIF: {args.output_file}")
    print(f"Кадров: {len(images)}, Длительность: {args.duration}мс, Повторов: {args.loop}")

if __name__ == "__main__":
    create_gif_from_args()
