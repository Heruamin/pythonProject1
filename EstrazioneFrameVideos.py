import cv2
import os

print(cv2.__version__)


def frame_capture(file, save_path):
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    print(vidcap)
    # print(image.shape)
    # global count
    count = 0
    success = True
    while success:
        cv2.imwrite(save_path % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
        print(count)

"""
Da aggiustare XD
"""

def main():
    path_video = "DataSet/UCF-101/ApplyEyeMakeup"
    video_format = ".avi"
    path_to_save = "hit%d.jpg"
    for file in os.listdir(path_video):
        if file.endswith(video_format):
            path = os.path.join(path_video, file)
            frame_capture(path, path_to_save)

input('stop2')
if __name__ == "__main__":
    input('stop')
    print("Iniziamo")
    main()
    print("Fine")

