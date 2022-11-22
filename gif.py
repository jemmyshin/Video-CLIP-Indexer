import glob
import os
from PIL import Image

# filepaths
fp_in = "/Users/jemfu/Desktop/Video-CLIP-Indexer/tmp_key_frames"
fp_out = "/Users/jemfu/Desktop"

frames = [Image.open(image) for image in glob.glob(f"{fp_in}/*")]
frame_one = frames[0]
frame_one.save("download.gif", format="GIF", append_images=frames,
           save_all=True, duration=1000, loop=1)

# import imageio
# images = []
# filenames = os.listdir(fp_in)
# for filename in filenames:
#     images.append(imageio.imread(os.path.join(fp_in, filename)))
# imageio.mimsave(os.path.join(fp_out, 'movie.gif'), images)