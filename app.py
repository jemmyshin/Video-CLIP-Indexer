import streamlit as st
from helper import search_frame
import os
from docarray import Document

st.set_page_config(page_title='Video CLIP Indexer', page_icon='🔍')
st.title('Video CLIP Indexer')
uploaded_file = st.file_uploader('Choose a file')
query = st.text_input('Text Query', '')
similarity_threshold = st.text_input('Similarity Threshold', '0.8')
cas_url = st.text_input('CLIP-as-service Server', 'grpc://0.0.0.0:51000')
token = st.text_input('Token', '<your access token>')
analysis_button = st.button('Extract Key Frames')
search_button = st.button('Search')

if analysis_button:
    with st.spinner('Extracting key frames...'):
        os.makedirs('tmp_videos', exist_ok=True)
        with open('tmp_videos/' + uploaded_file.name, 'wb') as f:
            f.write(uploaded_file.getvalue())

        d = Document(uri='tmp_videos/' + uploaded_file.name).load_uri_to_video_tensor(
            only_keyframes=False)

        st.session_state.original_video = d
        keyframes = [
            Document(tensor=d.tensor[i]).convert_image_tensor_to_blob()
            for i in range(len(d.tensor)) if i in d.tags['keyframe_indices']]
        for idx, frame in enumerate(keyframes):
            frame.tags['index'] = idx
        st.session_state.keyframes = keyframes
        st.success('Done extracting key frames, now you can search for it!')

if search_button:
    if 'keyframes' in st.session_state and 'original_video' in st.session_state:
        video = st.session_state.original_video

        with st.spinner(
                f"We are searching from {len(st.session_state.keyframes)} keyframes..."):
            tags, id, scores = search_frame(st.session_state.keyframes,
                                            query,
                                            cas_url, token)
            index = int(tags[0]['index'])
            most_similar_scene = Document(tensor=video.tensor[
                                                 video.tags['keyframe_indices'][
                                                     index]: min(len(video.tensor),
                                                                 video.tags[
                                                                     'keyframe_indices'][
                                                                     index + 1])
                                                 ])
            max_similarity_score = scores[0]

            if max_similarity_score >= float(similarity_threshold):
                os.makedirs('tmp_videos', exist_ok=True)
                most_similar_scene.save_video_tensor_to_file(file='tmp_videos/tmp.mp4')
                st.success(
                    f'Found a match with similarity score: {max_similarity_score}')
                st.video('tmp_videos/tmp.mp4')
                os.remove('tmp_videos/tmp.mp4')
            else:
                st.success(f'No match found. Max similarity score: {max_similarity_score} '
                           f'is smaller than threshold: {similarity_threshold}')
    else:
        st.warning('Please extract the key frame first')
