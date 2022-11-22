import numpy as np
import streamlit as st
from helper import search_frame, get_keyframes_data
import os
os.system('sudo apt-get install --no-upgrade ffmpeg')
import skvideo.io
import shutil
import av
from docarray import DocumentArray, Document


st.set_page_config(page_title='Video CLIP Indexer', page_icon='🔍')
st.title('Video CLIP Indexer')
uploaded_file = st.file_uploader('Choose a file')
text_prompt = st.text_input('Text Prompt', '')
topn_value = st.text_input('Top N', '5')
num_frames = st.text_input('Num_frames', '50')
cut_sim_value = st.text_input('Cut Sim', '0.6')
cas_url = st.text_input('CLIP-as-service Server', 'grpcs://api.clip.jina.ai:2096')
token = st.text_input('Token', '31454a8d0823445012c6de5623aed215')
analysis_button = st.button('Extract Key Frames')
search_button = st.button('Search')

if analysis_button:
    with st.spinner('Extracting key frames...'):
        os.makedirs('tmp_videos', exist_ok=True)
        os.makedirs('tmp_key_frames', exist_ok=True)
        with open('tmp_videos/' + uploaded_file.name, 'wb') as f:
            f.write(uploaded_file.getvalue())

        video_data = skvideo.io.vread('tmp_videos/' + uploaded_file.name)
        keyframe_data = get_keyframes_data(video_data, float(cut_sim_value))
        spams, ndarray, scores = search_frame(keyframe_data, text_prompt,
                                              int(topn_value), cas_url, token)
        for spam in spams:
            i = spams.index(spam)
            save_name = 'tmp_videos/' + str(i) + '_tmp.mp4'
            skvideo.io.vwrite(save_name,
                              video_data[int(spam['left']):int(spam['right'])])
            st.video(save_name)
            os.remove(save_name)
        st.success('Done extracting key frames, now you can search for it!')
        # keyframe_data = []
        # with av.open('tmp_videos/' + uploaded_file.name) as container:
        #     # Signal that we only want to look at keyframes.
        #     stream = container.streams.video[0]
        #     stream.codec_context.skip_frame = 'NONKEY'
        #     for frame in container.decode(stream):
        #         frame.to_image().save(
        #             'tmp_key_frames/frame.{:04d}.png'.format(frame.pts),
        #             quality=80,
        #         )
        #         keyframe_data.append('tmp_key_frames/frame.{:04d}.png'.format(frame.pts))
        # keyframe_data = keyframe_data[:5]
        # st.session_state.keyframe_data = DocumentArray([Document(id=tup.split('/')[1],
        #                                                          uri=tup).load_uri_to_blob() for tup in keyframe_data])
        # for d in st.session_state.keyframe_data:
        #     d.uri = None
        # shutil.rmtree('tmp_videos')
        # # shutil.rmtree('tmp_key_frames')
        # st.success('Done extracting key frames, now you can search for it!')

if search_button:
    if 'keyframe_data' in st.session_state:
        with st.spinner(f"We are searching from {len(st.session_state.keyframe_data)} frames..."):
            results = search_frame(st.session_state.keyframe_data, text_prompt, int(topn_value), cas_url, token)
            results = DocumentArray([doc.convert_blob_to_image_tensor() for doc in results])
            img = np.array([r.tensor for r in results])



            # os.makedirs('tmp_videos', exist_ok=True)
            # save_name = 'tmp_videos/tmp.mp4'
            #
            # container = av.open(save_name, 'w')
            # for im in img:
            #     frame = av.VideoFrame.from_ndarray(im, format="rgb24")
            #     for packet in stream.encode(frame):
            #         container.mux(packet)
            #
            # st.video(save_name)
            # os.remove(save_name)
            # shutil.rmtree('tmp_videos')

        st.success('Done!')
    else:
        st.warning('Please extract the key frame first')
