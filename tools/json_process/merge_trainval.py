import json
import os

json_path = '/data/cenzhaojun/detectron2/tools/json_process'
train_json = json_path + os.sep + 'instance_train.json'
val_json = json_path + os.sep + 'instance_val.json'

with open(val_json,) as f:
    data = json.load(f)
    print(type(data))
    # print(data)
    for index, anns in enumerate(data['annotations']):

        print(index)
        print(anns)
        # 修改json文件的数值，在val里面的每一个image_id里面都加上一个5000
        print(anns['image_id'])
        anns['image_id'] = anns['image_id'] + 5000
        print("修改后的")
        print(anns['image_id'])
        # 还有对anno里面的ID进行修改
        anns['id'] = anns['id'] + 41110

    for index, img in enumerate(data['images']):
        print(index)
        print(img)
        # 修改json文件中images的数值，在id上面都加一个5000
        print(img['id'])
        img['id'] = img['id'] + 5000
        print("修改后的")
        print(img['id'])
print(data)
# 将修改后的文件保存下来
with open("train_val.json","w") as f:
    json.dump(data,f,ensure_ascii=False)


    # for anns in data['annotations']:
    #     print(anns)
    # for image in data['images']:
    #     print(image)