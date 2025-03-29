'''
datasets:
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/LLaVA-ReCap-118K.json
    sampling_strategy: all
    data_root: https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-118K
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/LLaVA-ReCap-CC3M.json
    sampling_strategy: all
    data_root: https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-CC3M
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/LLaVA-ReCap-558K.json
    sampling_strategy: all
    data_root: https://huggingface.co/datasets/lmms-lab/LLaVA-ReCap-558K
  - json_path: OpenGVLab/stage2/LLaVA-OneVision-Mid-Data/evol_instruct/evol_instruct_processed.json
    sampling_strategy: all
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/video/webvid-fuse_caption_2m.json
    sampling_strategy: all
    data_root: https://github.com/m-bain/webvid
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/video/caption_sharegemini_webvid_core100k_clean.json
    sampling_strategy: all
    data_root: https://github.com/m-bain/webvid
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/video/caption_sharegemini_k400_223k.json
    sampling_strategy: all
    data_root: https://opendatalab.com/OpenMMLab/Kinetics-400
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/ureader_tr_processed.json
    data_root: https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data/tree/main/ureader_ur/
    sampling_strategy: all
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/synthdog_zh_processed.json
    data_root: https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data/tree/main/synthdog_zh/synthdog_zh_images/
    sampling_strategy: all
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/synthdog_en_processed.json
    data_root: https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data/tree/main/synthdog_en/synthdog_en_images/
    sampling_strategy: all
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/video/smit_caption_481k.json
    sampling_strategy: all
    data_root: http://moments.csail.mit.edu/spoken.html
  - json_path: OpenGVLab/VideoChat-Flash-Training-Data/annotations/video/caption_sharegptvideo_300k-sharegptvideo-train_300k_302k.json
    sampling_strategy: all
    data_root: https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/tree/main/train_300k
    video_read_type: img
'''

import json

# for json_path in [
#     'OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/LLaVA-ReCap-118K.json',
#     'OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/LLaVA-ReCap-CC3M.json',
#     'OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/LLaVA-ReCap-558K.json',
#     'OpenGVLab/stage2/LLaVA-OneVision-Mid-Data/evol_instruct/evol_instruct_processed.json',
#     'OpenGVLab/VideoChat-Flash-Training-Data/annotations/video/webvid-fuse_caption_2m.json',
#     'OpenGVLab/VideoChat-Flash-Training-Data/annotations/video/caption_sharegemini_webvid_core100k_clean.json',
#     'OpenGVLab/VideoChat-Flash-Training-Data/annotations/video/caption_sharegemini_k400_223k.json',
#     'OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/ureader_tr_processed.json',
#     'OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/synthdog_zh_processed.json',
#     'OpenGVLab/VideoChat-Flash-Training-Data/annotations/image/synthdog_en_processed.json',
#     'OpenGVLab/VideoChat-Flash-Training-Data/annotations/video/smit_caption_481k.json',
#     'OpenGVLab/VideoChat-Flash-Training-Data/annotations/video/caption_sharegptvideo_300k-sharegptvideo-train_300k_302k.json'
# ]:

# yaml path : ./data/stage3_short-long_mix_sft.yaml


import yaml, os
yaml_data = yaml.load(open('./data/stage3_short-long_mix_sft.yaml'), Loader=yaml.FullLoader)

n_good = 0
for instance in yaml_data['datasets']:
    json_path = instance['json_path']
    if not os.path.exists(json_path):
        print('Missing:',json_path)
    else:
        try:
            if json_path.endswith('.json'):
                n_good += 1
                data = json.load(open(json_path))
                print(json_path)
                print(data[0])
            elif json_path.endswith('.jsonl'):
                n_good += 1
                raw_data = open(json_path).readlines()
                data = [json.loads(line) for line in raw_data[:10]]
                print(json_path)
                print(data[0])
            else:
                print('Unknown:',json_path)
        except:
            print('Error:',json_path)
            n_good -= 1
            continue

print(n_good)