"""
Usage: python3 xml_to_csv.py [train|test]
"""

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member.find('name').text,
                     int(bndbox.find('xmin').text),
                     int(bndbox.find('ymin').text),
                     int(bndbox.find('xmax').text),
                     int(bndbox.find('ymax').text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    mode = sys.argv[1]	
    print(mode)
    image_path = os.path.join(os.getcwd(), mode)
    print('Obtaining descriptions from: ' + image_path)
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(mode + '_labels.csv', index=None)
    print('Successfully converted xml to csv.')


main()
