import xml.etree.ElementTree as ET
import os


def convert_cls_id(cls_id):
    mapping = {
        '100000001': 0,
        '100000002': 1,
        '100000003': 2,
        '100000004': 3,
        '100000005': 4,
        '100000006': 5,
        '100000007': 6,
        '100000008': 7,
        '100000009': 8,
        '100000010': 9,
        '100000011': 10,
        '100000012': 11,
        '100000013': 12,
        '100000015': 13,
        '100000016': 14,
        '100000017': 15,
        '100000018': 16,
        '100000019': 17,
        '100000020': 18,
        '100000022': 19,
        '100000024': 20,
        '100000025': 21,
        '100000026': 22,
        '100000027': 23,
        '100000028': 24,
        '100000029': 25,
        '100000030': 26,
        '100000032': 27,
    }
    return mapping[cls_id]


def xml_to_txt(xml_file_dir, txt_file_dir):
    os.makedirs(txt_file_dir, exist_ok=True)
    all_class_ids = set()
    for xml_file in os.listdir(xml_file_dir):
        if not xml_file.endswith('.xml') or xml_file == 'annotation_fmt.xml':
            continue
        xml_file_path = os.path.join(xml_file_dir, xml_file)
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        img_id = root.find('Img_ID').text
        img_width = int(root.find('Img_SizeWidth').text)
        img_height = int(root.find('Img_SizeHeight').text)
        txt_file_path = os.path.join(txt_file_dir, f"{img_id}.txt")
        objs = root.findall('.//HRSC_Object')
        if len(objs) == 0:
            continue
        with open(txt_file_path, 'w') as txt_file:
            for obj in objs:
                class_id = convert_cls_id(obj.find('Class_ID').text)
                all_class_ids.add(class_id)
                box_xmin = int(obj.find('box_xmin').text)
                box_ymin = int(obj.find('box_ymin').text)
                box_xmax = int(obj.find('box_xmax').text)
                box_ymax = int(obj.find('box_ymax').text)
                x_center = ((box_xmin + box_xmax) / 2) / img_width
                y_center = ((box_ymin + box_ymax) / 2) / img_height
                box_width = (box_xmax - box_xmin) / img_width
                box_height = (box_ymax - box_ymin) / img_height
                txt_file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
    print(f"Total class ids: {len(all_class_ids)}")
    print([str(class_id) for class_id in all_class_ids])


def main():
    xml_file_dir = r"H:\Library\Datasets\HRSC\HRSC2016_dataset\HRSC2016\FullDataSet\Annotations"
    txt_file_dir = r"H:\Library\Datasets\HRSC\HRSC2016_dataset\HRSC2016\FullDataSet-YOLO\Annotations"
    xml_to_txt(xml_file_dir, txt_file_dir)


if __name__ == '__main__':
    main()
