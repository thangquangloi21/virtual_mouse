from moviepy.editor import VideoFileClip, vfx
import os

input_videos = "D:\\Videos"
output_videos = "D:\\All_learn_programs\\Python\\virtual_mouse\\Video_test"

# Ensure output directory exists
os.makedirs(output_videos, exist_ok=True)

for file in os.listdir(input_videos):
    if file.endswith(".mp4"):
        input_path = os.path.join(input_videos, file)
        output_path = os.path.join(output_videos, os.path.splitext(file)[0] + "_mirrored.mp4")

        # Load video and apply effect
        clip = VideoFileClip(input_path)
        reversed_clip = clip.fx(vfx.mirror_x)

        # Write the output file
        reversed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Close the clip to free resources
        clip.close()
        reversed_clip.close()



