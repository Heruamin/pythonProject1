from moviepy.editor import VideoFileClip

clip = VideoFileClip("DataSet/UCF-CRIME/Fighting/Fighting002_x264.mp4")
print(int(clip.fps))
print(int(clip.duration))