import cv2
import os

"""
Serve avere la cartella DataSet con la struttura :
   - UCF101 -> folder -> video.avi
   - UT-Interaction -> folder -> video.avi   
"""


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


def ucf101():
    # Siamo nel dataset UFC101
    DataSet_UFC101 = "DataSet/UCF-101"
    for class_video in os.listdir(DataSet_UFC101):
        path_video = os.path.join(DataSet_UFC101, class_video)  # Esempio "DataSet/UCF-101/ApplyEyeMakeup"
        # Count serve per poter inserire tutti i frame senza sovrascizioni
        count = 0
        video_format = ".avi"  # Abbastanza inutile questo controllo, ma non si sa mai
        for file in os.listdir(path_video):
            if file.endswith(video_format):
                # Ogni file video genererà una cartella con lo stesso suo nome con all'interno i suoi frame.
                id_video = file.split("_")[2]
                n_clip = file.split("_")[3].split(".")[0]
                if n_clip == "c01":
                    count = 0
                folder_name = file[0:len(file) - len(video_format + "_c00")]
                # creo la cartella
                os.makedirs(os.path.join(path_video, folder_name), exist_ok=True)

                path_to_save = os.path.join(path_video, folder_name, id_video + "_frame%d.jpg")
                path = os.path.join(path_video, file)
                count = frame_capture(path, path_to_save, count)


def utinteraction():
    # Siamo nel dataset UFC101
    DataSet_UTInteraction = "DataSet/UT-Interaction"
    for class_video in os.listdir(DataSet_UTInteraction):
        path_video = os.path.join(DataSet_UTInteraction, class_video)  # Esempio "DataSet/UT-Interaction/hand shake"
        # Count serve per poter inserire tutti i frame senza sovrascizioni
        count = 0
        video_format = ".avi"  # Abbastanza inutile questo controllo, ma non si sa mai
        for file in os.listdir(path_video):
            if file.endswith(video_format):
                # Ogni file video genererà una cartella con lo stesso suo nome con all'interno i suoi frame.
                count = 0
                nome = "_".join(file.split("_")[0:2])
                folder_name = file[0:len(file) - len(video_format)]
                # creo la cartella
                os.makedirs(os.path.join(path_video, folder_name), exist_ok=True)

                path_to_save = os.path.join(path_video, folder_name, nome + "frame%d.jpg")
                path = os.path.join(path_video, file)
                _ = frame_capture(path, path_to_save, count)


def hmdb51():
    """
    To-Do
    """
    return True


if __name__ == "__main__":
    print("Iniziamo")
    utinteraction()
    ucf101()
    hmdb51()
    print("Fine")
