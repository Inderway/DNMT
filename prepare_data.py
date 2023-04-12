# prepare the data: turn the raw data into json format
# created by wei
# Mar 27, 2023

import json
import os

datatype='sent' #sent if sentence
img_path=f"data/{datatype}/image"
# literal translation
ltr_path=f"data/{datatype}/text_l"
# free translation
fre_path=f"data/{datatype}/text_f"

dataset_path=f'data/dongba_{datatype}.json'
data=[]
# write json
with open(dataset_path,'w',encoding='utf-8') as f:
    if datatype=='para':
        img_names=os.listdir(img_path)
        ltr_names=os.listdir(ltr_path)
        img_names.sort()
        ltr_names.sort()
        for i in range(len(img_names)):
            term={}
            term['document']=img_names[i].split('_')[0]
            term['file_name']=img_names[i]
            term['dongba']=os.path.join(img_path,img_names[i])
            with open(os.path.join(ltr_path,ltr_names[i]),'r',encoding='utf-8') as ltr:
                ltr_text=ltr.read()
                term['literal_translation']=ltr_text
            with open(os.path.join(fre_path,ltr_names[i]),'r',encoding='utf-8') as fre:
                fre_text=fre.read()
                term['free_translation']=fre_text
            data.append(term)
        json.dump(data, f, ensure_ascii=False)
    if datatype=='sent':
        doc_names=os.listdir(ltr_path)
        for name in doc_names:
            with open(os.path.join(ltr_path, name), 'r', encoding='utf-8') as ltr:
                ltr_text=ltr.readlines()
            with open(os.path.join(fre_path, name), 'r', encoding='utf-8') as fre:
                fre_text=fre.readlines()
            for i in range(len(ltr_text)):
                data.append([ltr_text[i].rstrip('\n'),fre_text[i].rstrip('\n')])
        json.dump(data, f, ensure_ascii=False)    
