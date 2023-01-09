import streamlit as st
from helper import search_frame
import os, shutil
from docarray import Document

st.set_page_config(page_title='Video CLIP Indexer', page_icon='🔍')
st.title('Video CLIP Indexer')
uploaded_file = st.file_uploader('Choose a file')
query = st.text_input('Text Query', '')
top_n = st.text_input('Top N', '5')
save_keyframes = st.selectbox('save keyframes', ('True', 'False'))
keyframes_dir = st.text_input('directory for saving keyframes', '')
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
        # keyframes = [
        #     Document(tensor=d.tensor[i]).convert_image_tensor_to_blob()
        #     for i in range(len(d.tensor)) if i in d.tags['keyframe_indices']]
        # for idx, frame in enumerate(keyframes):
        #     frame.tags['index'] = idx

        if save_keyframes:
            shutil.rmtree(keyframes_dir)
            os.makedirs(keyframes_dir, exist_ok=True)

        keyframes = []
        for i in range(len(d.tensor)):
            if i in d.tags['keyframe_indices']:
                keyframe = Document(tensor=d.tensor[i],
                                    tags={'index': len(keyframes)})
                if save_keyframes:
                    keyframe.save_image_tensor_to_file(file=f'{keyframes_dir}/{len(keyframes)}.png')
                keyframe.convert_image_tensor_to_blob()
                keyframes.append(keyframe)

        st.session_state.keyframes = keyframes
        st.success('Done extracting key frames, now you can search for it!')

if search_button:
    if 'keyframes' in st.session_state and 'original_video' in st.session_state:
        video = st.session_state.original_video

        with st.spinner(
                f"We are searching from {len(st.session_state.keyframes)} keyframes..."):
            tags, id, scores = search_frame(st.session_state.keyframes,
                                            query, int(top_n),
                                            cas_url, token)
            max_similarity_score = scores[0]
            if max_similarity_score < float(similarity_threshold):
                st.success(
                    f'No match found. Max similarity score: {max_similarity_score} '
                    f'is smaller than threshold: {similarity_threshold}')

            for i in range(len(id)):
                index = int(tags[i]['index'])
                start_index = video.tags['keyframe_indices'][index]
                end_index = len(video.tensor) if index == len(
                    video.tags['keyframe_indices']) - 1 else \
                video.tags['keyframe_indices'][index + 1]

                similar_scene = Document(tensor=video.tensor[start_index: end_index])
                similarity_score = scores[i]

                if similarity_score >= float(similarity_threshold):
                    os.makedirs('tmp_videos', exist_ok=True)
                    similar_scene.save_video_tensor_to_file(
                        file='tmp_videos/tmp.mp4')
                    st.text(
                        f'Top {i + 1} match -- similarity score: {similarity_score}')
                    st.video('tmp_videos/tmp.mp4')
                    os.remove('tmp_videos/tmp.mp4')
    else:
        st.warning('Please extract the key frame first')
