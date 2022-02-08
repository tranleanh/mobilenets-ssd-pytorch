import os
import os.path as osp

import json

from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom

from PIL import Image

from tqdm import tqdm

DEBUG = False

BDD_FOLDER = osp.join("..", "bdd100k", "bdd100k")

if DEBUG:
    XML_PATH = osp.join(".", "xml")
else:
    XML_PATH = osp.join(BDD_FOLDER, "xml")


def bdd_to_voc(bdd_folder, xml_folder):
    """
    Convert BDD100k json file to PASCAL VOC stype xml files.
    :param bdd_folder: a path to bdd100k which contains images, labels folder.
    :param xml_folder: a path to save the xml files.
    :return:
    """
    image_path = osp.join(bdd_folder, "images", "100k", "%s")
    label_path = osp.join(bdd_folder, "labels", "bdd100k_labels_images_%s.json")

    classes = set()

    # for trainval in ['val']:
    for trainval in ['train', 'val']:
        image_folder = image_path % trainval
        json_path = label_path % trainval
        xml_folder_ = osp.join(xml_folder, trainval)

        if not os.path.exists(xml_folder_):
            os.makedirs(xml_folder_)

        with open(json_path) as f:
            j = f.read()
        data = json.loads(j)

        for datum in tqdm(data):
            annotation = Element('annotation')
            SubElement(annotation, 'folder').text = trainval
            SubElement(annotation, 'filename').text = datum['name']
            size = get_size(osp.join(image_folder, datum['name']))
            annotation.append(size)

            # additional information
            for key, item in datum['attributes'].items():
                SubElement(annotation, key).text = item

            # bounding box
            for label in datum['labels']:
                try:
                    box2d = label['box2d']
                except KeyError:
                    continue
                else:
                    bndbox = get_bbox(box2d)

                object_ = Element('object')
                SubElement(object_, 'name').text = label['category']
                classes.add(label['category'])

                # additional information
                for key, item in label['attributes'].items():
                    if type(item) == str:
                        SubElement(object_, key).text = item
                    elif type(item) == bool:
                        SubElement(object_, key).text = '1' if item else '0'
                    elif type(item) in [int, float]:
                        SubElement(object_, key).text = str(item)
                    else:
                        raise ValueError("%s could not be handled" % type(item))

                object_.append(bndbox)
                annotation.append(object_)

            xml_filename = osp.splitext(datum['name'])[0] + '.xml'
            with open(osp.join(xml_folder_, xml_filename), 'w') as f:
                f.write(prettify(annotation))
    print(classes)


def get_size(image_path):
    i = Image.open(image_path)
    sz = Element('size')
    SubElement(sz, 'width').text = str(i.width)
    SubElement(sz, 'height').text = str(i.height)
    SubElement(sz, 'depth').text = str(3)
    return sz


def get_bbox(box2d):
    bndbox = Element('bndbox')
    SubElement(bndbox, 'xmin').text = str(round(box2d['x1']))
    SubElement(bndbox, 'ymin').text = str(round(box2d['y1']))
    SubElement(bndbox, 'xmax').text = str(round(box2d['x2']))
    SubElement(bndbox, 'ymax').text = str(round(box2d['y2']))
    return bndbox


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
        """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


if __name__ == "__main__":
    bdd_to_voc(BDD_FOLDER, XML_PATH)
