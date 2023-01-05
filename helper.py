from clip_client import Client
from docarray import Document, DocumentArray
from typing import Union, List


def search_frame(keyframe_da: Union[List[Document], DocumentArray], query: str,
                 top_n: int, server_url: str, token: str):
    client = Client(server_url, credential={'Authorization': token})
    d = Document(text=query, matches=keyframe_da)
    r = client.rank([d], show_progress=True)
    result = r['@m', ['tags', 'id', 'scores__clip_score__value']]
    return [item[:top_n] for item in result]

