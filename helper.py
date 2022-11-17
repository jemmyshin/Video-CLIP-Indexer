from clip_client import Client
from docarray import Document, DocumentArray
import imagehash
from PIL import Image
import numpy as np
import os


keyframe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keyframes')
if not os.path.exists(keyframe_path):
    os.mkdir(keyframe_path)

def get_keyframes_data(video_data: 'np.ndarray', cut_sim: float):
    last_hash = imagehash.phash(Image.fromarray(video_data[0]))
    key_frames = [0]
    frame_num = 0
    for each_frame in video_data:
        frame_hash = imagehash.phash(Image.fromarray(each_frame))
        similarity = 1 - (last_hash - frame_hash) / len(frame_hash.hash) ** 2
        if similarity < cut_sim:
            key_frames.append(frame_num)
        frame_num += 1
        last_hash = frame_hash
    video_length = len(video_data)
    key_frames.append(video_length)
    keyframes_data = [((i, key_frames[key_frames.index(i)+1]), video_data[i]) for i in key_frames if i != video_length]
    
    if os.path.exists(keyframe_path):
        os.system('cd {} && rm *'.format(keyframe_path))
    for i, keyframe in enumerate(keyframes_data):
        Image.fromarray(keyframe[1]).save(f"{keyframe_path}/{i}.jpeg")

    return keyframes_data


def search_frame(keyframe_da: DocumentArray, prompt: str, topn: int, server_url: str):
    client = Client(server_url, credential={'Authorization': os.getenv('JINA_AUTH_TOKEN')})
    d = Document(text=prompt, matches=keyframe_da)
    r = client.rank([d], show_progress=True)
    result = r['@m', ['tags', 'blob', 'scores__clip_score__value']]
    return [each[:topn] for each in result]
