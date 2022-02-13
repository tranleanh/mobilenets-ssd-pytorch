import glob
import os

os.path.join("..")
img_src = glob.glob("../bdd100k/bdd100k/images/100k/train/*.jpg")
xml_src = glob.glob("../bdd100k/bdd100k/xml/train/*.xml")

img_name = []

num_img = 0
for img in img_src:
    num_img += 1
    img_basename = os.path.basename(img)
    img_onlyname = os.path.splitext(img_basename)

    img_name.append(img_onlyname[0])
    
print(num_img)

xml_name = []

num_xml = 0
for xml in xml_src:
    num_xml += 1
    xml_basename = os.path.basename(xml)
    xml_onlyname = os.path.splitext(xml_basename)

    xml_name.append(xml_onlyname[0])
    
print(num_xml)

not_in_list = []

for img in img_name:
    if img not in xml_name: not_in_list.append(img)

print(len(not_in_list))
print(not_in_list)

path = "../bdd100k/bdd100k/images/100k/train/" + not_in_list[0] + ".jpg"
print(path)

# Remove training samples which do not have anotation.
count = 0
for item in not_in_list:
    path = "../bdd100k/bdd100k/images/100k/train/" + item + ".jpg"
    if os.path.exists(path):
        os.remove(path)
        count += 1
    else:
        print("The file does not exist")
        
print(count)