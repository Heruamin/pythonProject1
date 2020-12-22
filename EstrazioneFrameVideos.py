import cv2
import os

print(cv2.__version__)


def frame_capture(file, save_path, count):
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    # print(vidcap)
    # print(image.shape)
    # global count
    # count = 0
    success = True
    while success:
        cv2.imwrite(save_path % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1
        # print(count)
    return count

"""
Da aggiustare XD
"""

def main_UCF101():
    # Siamo nel dataset UFC101
    DataSet_UFC101 = "DataSet/UCF-101"
    for class_video in os.listdir(DataSet_UFC101):
        path_video = os.path.join(DataSet_UFC101, class_video) # Esempio "DataSet/UCF-101/ApplyEyeMakeup"
        # Count serve per poter inserire tutti i frame senza sovrascizioni
        count = 0
        initial_id_video = 'g01'
        video_format = ".avi" # Abbastanza inutile questo controllo, ma non si sa mai
        for file in os.listdir(path_video):
            if file.endswith(video_format):
                # Ogni file video genererà una cartella con lo stesso suo nome con all'interno i suoi frame.
                id_video = file.split("_")[2]
                n_clip = file.split("_")[3].split(".")[0]
                if n_clip == "c01":
                    count = 0
                folder_name = file[0:len(file)-len(video_format + "_c00")]
                # creo la cartella
                os.makedirs(os.path.join(path_video,folder_name), exist_ok = True)

                path_to_save = os.path.join(path_video,folder_name,id_video + "_frame%d.jpg")
                path = os.path.join(path_video, file)
                count = frame_capture(path, path_to_save, count)

input('stop2')
if __name__ == "__main__":
    input('stop')
    print("Iniziamo")
    main_UCF101()
    print("Fine")

