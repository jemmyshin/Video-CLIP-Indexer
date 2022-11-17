import streamlit as st
from helper import search_frame, get_keyframes_data
import os
import skvideo.io
from docarray import DocumentArray, Document


st.set_page_config(page_title='Video CLIP Indexer', page_icon='🔍')
st.title('Video CLIP Indexer')
uploaded_file = st.file_uploader('Choose a file')
text_prompt = st.text_input('Text Prompt', '')
topn_value = st.text_input('Top N', '5')
num_frames = st.text_input('Num_frames', '50')
cut_sim_value = st.text_input('Cut Sim', '0.6')
cas_url = st.text_input('CLIP-as-service Server', 'grpcs://api.clip.jina.ai:2096')
search_button = st.button('Search')
analysis_button = st.button('Analysis')

if analysis_button:
    with st.spinner('Processing...'):
        os.makedirs('tmp_videos', exist_ok=True)
        with open('tmp_videos/' + uploaded_file.name, 'wb') as f:
            f.write(uploaded_file.getvalue())
        st.session_state.video_data = skvideo.io.vread('tmp_videos/' + uploaded_file.name, num_frames=num_frames)
        keyframe_data = get_keyframes_data(st.session_state.video_data, float(cut_sim_value))
        st.session_state.keyframe_data = DocumentArray([Document(tags={'left': str(tup[0][0]), 'right': str(tup[0][1])}, tensor=tup[1]).convert_image_tensor_to_blob() for tup in keyframe_data])
        for d in st.session_state.keyframe_data:
            d.tensor = None

if search_button:
    if 'keyframe_data' in st.session_state:
        with st.spinner(f"We are searching from {len(st.session_state.keyframe_data)} frames..."):
            spams, ndarray, scores = search_frame(st.session_state.keyframe_data, text_prompt, int(topn_value), cas_url)
            for spam in spams:
                i = spams.index(spam)
                save_name = 'tmp_videos/' + str(i) + '_tmp.mp4'
                skvideo.io.vwrite(save_name, st.session_state.video_data[int(spam['left']):int(spam['right'])])
                st.video(save_name)
                os.remove(save_name)
        st.success('Done!')
