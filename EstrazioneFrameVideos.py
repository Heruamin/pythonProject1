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
    # Inseriamo i nomi delle cartelle che finiranno in violence
    list_violence = ["Punch", "SumoWrestling"]
    # Inseriamo i nomi delle cartelle che finiranno in non_violence
    list_noviolence = ["Basketball", "Skiing"]
    # Path del dataset
    DataSet_UFC101 = "DataSet/UCF-101"
    # Path di salvataggio dei frame
    DataExtract_UFC101 = "DataExtract/UCF-101"
    # Classe dei video : Violence o NoViolence
    class_folder = " "

    for class_video in os.listdir(DataSet_UFC101):
        if class_video in list_violence:
            class_folder = "Violence"
        elif class_video in list_noviolence:
            class_folder = "NoViolence"
        else:
            continue
        path_video = os.path.join(DataSet_UFC101, class_video)  # Esempio "DataSet/UCF-101/ApplyEyeMakeup"
        # Count serve per poter inserire tutti i frame senza sovrascizioni
        count = 0
        video_format = ".avi"  # Abbastanza inutile questo controllo, ma non si sa mai
        for file in os.listdir(path_video):
            if file.endswith(video_format):
                # Ogni file video genererà una cartella con lo stesso suo nome con all'interno i suoi frame.
                id_video = file.split("_")[2]
                n_clip = file.split("_")[3].split(".")[0]
                # Questo controllo serve per poter ripartire con count = 0 se cambiamo sequenza di clip
                if n_clip == "c01":
                    count = 0

                folder_name = file[0:len(file) - len(video_format + "_c00")]
                # creo la cartella
                # os.makedirs(os.path.join(path_video, folder_name), exist_ok=True)
                # la riga sottostante serve per creare le cartelle per ogni video, ma non vogliamo questo ora
                # path_to_save = os.path.join(path_video, folder_name, id_video + "_frame%d.jpg")
                # Creo la cartella violence e noviolence e se ci sono allora no_problem
                os.makedirs(os.path.join(DataExtract_UFC101, class_folder), exist_ok=True)
                # i diversi frame di ogni video saranno distinguibili
                path_to_save = os.path.join(DataExtract_UFC101, class_folder, folder_name + "_frame%d.jpg")
                # questo è il path del file video
                path_file = os.path.join(path_video, file)

                count = frame_capture(path_file, path_to_save, count)


def utinteraction():
    # Siamo nel dataset UT-Interaction
    # Inseriamo i nomi delle cartelle che finiranno in violence
    list_violence = ["kick", "push", "punching"]
    # Inseriamo i nomi delle cartelle che finiranno in non_violence
    list_noviolence = ["hand shake", "hugging", "pointting"]
    # Ci sono meno commenti perchè la logica è simile a quella della funzione ucf101
    DataSet_UTInteraction = "DataSet/UT-Interaction"
    DataExtract_UTInteraction = "DataExtract/UT-Interaction"
    class_folder = " "
    for class_video in os.listdir(DataSet_UTInteraction):
        if class_video in list_violence:
            class_folder = "Violence"
        elif class_video in list_noviolence:
            class_folder = "NoViolence"
        else:
            continue
        path_video = os.path.join(DataSet_UTInteraction, class_video)  # Esempio "DataSet/UT-Interaction/hand shake"
        # Count serve per poter inserire tutti i frame senza sovrascizioni
        count = 0
        video_format = ".avi"  # Abbastanza inutile questo controllo, ma non si sa mai
        for file in os.listdir(path_video):
            if file.endswith(video_format):
                # Ogni file video genererà una cartella con lo stesso suo nome con all'interno i suoi frame.
                count = 0
                # Prendo solo i primi due numeri ##_##
                nome = "_".join(file.split("_")[0:2])
                # potrebbe essere inutile
                folder_name = file[0:len(file) - len(video_format)]
                # creo la cartella e se esiste no problem
                os.makedirs(os.path.join(DataExtract_UTInteraction, class_folder), exist_ok=True)
                # os.makedirs(os.path.join(path_video, folder_name), exist_ok=True)

                path_to_save = os.path.join(DataExtract_UTInteraction, class_folder, nome + "frame%d.jpg")
                # path del video
                path_file = os.path.join(path_video, file)
                # in questo caso la count viene sempre fatta partire da 0 poichè sono clip diverse
                _ = frame_capture(path_file, path_to_save, count)


# Questa funzione lavora un po' diversamente dalle precedenti perchè si occuperà di prendere i frame, rinominarli
# e spostarli
def hmdb51():
    # Siamo nel dataset HMDB51
    # Inseriamo i nomi delle cartelle che finiranno in violence
    list_violence = ["kick", "sword", "shoot_gun", "punch", "hit"]
    # Inseriamo i nomi delle cartelle che finiranno in non_violence
    list_noviolence = ["walk", "sit", "shake_hands", "pick", "wave"]
    # Ci sono meno commenti perchè la logica è simile a quella della funzione ucf101
    DataSet_HMDB51 = "DataSet/HMDB51"
    DataExtract_HMDB51 = "DataExtract/HMDB51"
    class_folder = " "
    for class_video in os.listdir(DataSet_HMDB51):  # (Esempio class_video = brush_hair )
        if class_video in list_violence:
            class_folder = "Violence"
        elif class_video in list_noviolence:
            class_folder = "NoViolence"
        else:
            continue
        folder_video_path = os.path.join(DataSet_HMDB51, class_video)  # DataSet/HMDB51/brush_hair
        for folder_frame in os.listdir(folder_video_path):  # folder_name = April_09_..._goo_0
            # creamo il nome del video             
            video_name = folder_frame

            # creamo le cartelle Violence e NoViolence, se esitono no problem
            os.makedirs(os.path.join(DataExtract_HMDB51, class_folder), exist_ok=True)
            # path cartella
            path_frames = os.path.join(folder_video_path, folder_frame)

            path_to_save = os.path.join(DataExtract_HMDB51, class_folder)

            for frames in os.listdir(path_frames):  # sono nella cartella con il folder_name con i jpg
                if not frames.endswith(".jpg"):
                    continue
                path_file = os.path.join(path_frames, frames)
                numero_frame = frames[1:5]
                path_to_save_file = os.path.join(path_to_save, video_name + "_" + numero_frame + ".jpg")

                os.rename(path_file, path_to_save_file)


def ucfcrime():
    # Siamo nel dataset UFC-CRIME
    # Inseriamo i nomi delle cartelle che finiranno in violence
    # i ragazzi non hanno messo ABUSE
    list_violence = ["Abuse", "Assault", "Fighting", "RoadAccidents", "Shoplifting",
                     "Arrest", "Burglary", "Robbery", "Stealing",
                     "Arson", "Explosion", "Shooting", "Vandalism"]
    # Inseriamo i nomi delle cartelle che finiranno in non_violence
    list_noviolence = ["Normal_Videos_event"]
    # Path del dataset
    DataSet_UFCRIME = "DataSet/UCF-CRIME"
    # Path di salvataggio dei frame
    DataExtract_UFCRIME = "DataExtract/UCF-CRIME"
    # Classe dei video : Violence o NoViolence
    class_folder = " "

    for class_video in os.listdir(DataSet_UFCRIME):
        if class_video in list_violence:
            class_folder = "Violence"
        elif class_video in list_noviolence:
            class_folder = "NoViolence"
        else:
            continue
        path_video = os.path.join(DataSet_UFCRIME, class_video)  # Esempio "DataSet/UCF-Crime/Abuse"
        # Count serve per poter inserire tutti i frame senza sovrascizioni
        count = 0
        video_format = ".mp4"  # Abbastanza inutile questo controllo, ma non si sa mai
        for file in os.listdir(path_video):
            if file.endswith(video_format):
                # Ogni file video genererà una cartella con lo stesso nome e all'interno i suoi frame
                # id_video = file.split("_")[2]
                nome_clip = file.split("_")[0]
                count = 0
                # creo la cartella
                # os.makedirs(os.path.join(path_video, folder_name), exist_ok=True)
                # la riga sottostante serve per creare le cartelle per ogni video, ma non vogliamo questo ora
                # path_to_save = os.path.join(path_video, folder_name, id_video + "_frame%d.jpg")
                # Creo la cartella violence e noviolence e se ci sono allora no_problem
                os.makedirs(os.path.join(DataExtract_UFCRIME, class_folder), exist_ok=True)
                # i diversi frame di ogni video saranno distinguibili
                path_to_save = os.path.join(DataExtract_UFCRIME, class_folder, nome_clip + "_frame%d.jpg")
                # questo è il path del file video
                path_file = os.path.join(path_video, file)

                _ = frame_capture(path_file, path_to_save, count)


if __name__ == "__main__":
    print("Iniziamo")
    print("UT-Interaction")
    utinteraction()
    print("UCF101")
    ucf101()
    print("HMDB51")
    hmdb51()
    print("UCFCRIME")
    ucfcrime()
    print("Fine")
    print("-----------------")
