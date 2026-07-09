import json
import os
from tqdm import tqdm

def coco_to_yolo(json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # image_id -> image_info
    images = {img['id']: img for img in data['images']}

    # category_id -> class_id（重新映射成0-N）
    categories = sorted(data['categories'], key=lambda x: x['id'])
    cat_id_map = {cat['id']: i for i, cat in enumerate(categories)}

    # image_id -> annotations
    img_ann_map = {}
    for ann in data['annotations']:
        img_ann_map.setdefault(ann['image_id'], []).append(ann)

    for img_id, img_info in tqdm(images.items()):
        file_name = img_info['file_name']
        w, h = img_info['width'], img_info['height']

        yolo_lines = []

        if img_id in img_ann_map:
            for ann in img_ann_map[img_id]:
                if ann.get('iscrowd', 0) == 1:
                    continue

                x, y, bw, bh = ann['bbox']

                # COCO bbox -> YOLO
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                bw /= w
                bh /= h

                class_id = cat_id_map[ann['category_id']]

                yolo_lines.append(f"{class_id} {x_center} {y_center} {bw} {bh}")

        # 写入txt
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)

        with open(txt_path, 'w') as f:
            f.write("\n".join(yolo_lines))


if __name__ == "__main__":
    json_path = "/home/fwq207/zzh/Multimodal_data/DroneVehicle/DV/rgb_test.json"   # 改成你的json路径
    output_dir = "/home/fwq207/zzh/Multimodal_data/DroneVehicle/DV/yolo_labels"       # 输出目录

    coco_to_yolo(json_path, output_dir)
    print("转换完成！")
