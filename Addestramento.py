"""
3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D
3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D
3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D
3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D
"""

#Facciamo tutte le import
import tensorflow.compat.v1 as tf
from keras.models import Model
from keras.utils import Sequence
from tensorflow.keras.layers import Dense, LSTM, Attention, Flatten, Input
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import numpy as np
from glob import glob  # tool to match file patterns
# clear the GPU memory in case it's loaded up from a previous experiment
from keras import backend as K

K.clear_session()


class IteratorMocap(Sequence):
    """
    Questo oggetto ci è utile per non sovracaricare la RAM ed iterare tutto ciò che c'è nel path passatogli
    """
    def __init__(self, path_list, batch_size):
        """
        :param self:
        :param path_list: Lista di path da controllare, con strutttura DataSet/Classe/File.quello che ti serve
        :param batch_size: Non usato in questa implementazione, ma si può usare
        :return:
        """
        if isinstance(path_list, list):
            self.lista_path = path_list
            # lunghezza è il numero di path che sono stati inviati
            self.lunghezza = len(path_list)
            self.count = 0
            self.batch_size = batch_size
        else:
            raise ValueError("Iteratore inizializzato male")

    def __iter__(self):
        """
        Serve solo come metodo per creare un iteratore
        :param self:
        :return:
        """
        return self

    def __next__(self):
        """
        Serve solo perchè è un metodo astratto da implementare
        :param self:
        :return:
        """
        return

    def __getitem__(self, count):
        """
        :param self:
        :param count: Indice che viene inviato per prendere i dati. Si basa sulla batch_size inviata al modello
        :return:
        """
        path = self.lista_path[count]
        file = np.load(path)
        y = path.split("/")[-2]

        if y != "NoViolence" and y != "Violence":
            raise ValueError(
                "L'elemento analizzato è diverso da Violence o NoViolence, probabilmente non devi prendere l'elemento in posizione -2, y : " + y)
        else:
            y_da_restituire = np.array([[0.]]) if y == "NoViolence" else np.array([[1.]])
            return tf.convert_to_tensor(file, np.float32), tf.convert_to_tensor(y_da_restituire, np.float32)

    def __len__(self):
        """
        Serve alla rete per sapere il numero di batch per epoca
        :param self:
        :return:
        """
        # E' il numero di batch per epoca
        # Ritorniamo la lunghezza poichè diamo un elemento alla volta
        lunghezza = self.lunghezza
        return lunghezza
# Caricamento dei dati, sto seguendo quello che ci ha mostrato vincenzo in :
# - https://colab.research.google.com/drive/1d7rSfAKhDoYBqX_2Rzj9LuznYEM9kiET?usp=sharing#scrollTo=Itd-h9N_ndB-

# Serve più avanti per il one_hot_encoding, ma potenzialmente eliminabile
n_classes = 2
# Nome del dataset che vogliamo usare per l'addestramento
nome_dataset = 'UCF-CRIME'
# Path relativo del dataset, c'è scritto drive perchè in origine si sarebbe dovuto trovare nel CoLab
path_drive = 'DataMocap3D/' + nome_dataset
# Prendiamo la lista di percorsi dei file, a noi servono i .npy
NoViol = glob(path_drive + '/NoViolence/*.npy')
Viol = glob(path_drive + '/Violence/*.npy')

# Costruiamo la lista delle etichette di classe
labels = []

for i in NoViol:
    labels.append(0)  # 0 = NoViol

for i in Viol:
    labels.append(1)  # 1 = Viol

labels = np.uint8(labels)
filenames = NoViol + Viol

# Numero di etichette e quindi di campioni
n_sample = len(labels)

print('Created list with labels. There are', n_sample, 'of them.')

# Prima facciamo il one_hot_encoding delle labels

labels_one_hot = to_categorical(labels, num_classes=n_classes)

print('First label (NoViolence, one-hot encodings):', labels[0], labels_one_hot[0])
print('Last label (Violence, one-hot encodings):', labels[-1], labels_one_hot[-1])

print('Labels shape in one-hot encoding:', labels_one_hot.shape)

data_train_temp, data_test, labels_train_temp, labels_test = train_test_split(filenames, labels_one_hot, test_size=0.3,
                                                                              random_state=17, shuffle=False)

# Printiamo le shape per sicurezza
print('Training_temp shape:', len(data_train_temp))
print('Training_temp labels shape:', len(labels_train_temp))

print('Test  shape:', len(data_test))
print('Test labels shape:', len(labels_test))

data_train, data_validation, labels_train, label_validation = train_test_split(data_train_temp, labels_train_temp,
                                                                               test_size=0.2, random_state=13,
                                                                               shuffle=False)
# Printiamo le shape per sicurezza
print('Training  shape:', len(data_train))
print('Training labels shape:', len(labels_train))

print('Validation  shape:', len(data_validation))
print('Validation labels shape:', len(label_validation))

# Creamo gli iteratori per l'addestramento
obj_iteratore_train = IteratorMocap(data_train, 10)
iteratore_train = iter(obj_iteratore_train)

obj_iteratore_validation = IteratorMocap(data_validation, 10)
iteratore_validation = iter(obj_iteratore_validation)

# Costruiamo il modello
# Questo è il nome con cui salveremo il modello
nome_modello = 'LSTM10'
# Costruzione del modello
inp = Input((10, 6890 * 3))
lstm = LSTM(128, input_shape=(10, 10 * 6890 * 3), return_sequences=True)(inp)
query = Dense(60)(lstm)
values = Dense(60)(lstm)
att = Attention()([query, values])
flat = Flatten()(att)
dense = Dense(30, activation='sigmoid')(flat)
end = Dense(2, activation='sigmoid')(dense)
model = Model(inputs=inp, outputs=end, name="Lstm_Attention")
# Compilazione del modello
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
# Guardiamo un rapido sommario
print(model.summary())

#Addestramento del modello
model.fit_generator(obj_iteratore_train, validation_data=obj_iteratore_validation, epochs=10)
# Salviamo il modello
model.save('/content/drive/MyDrive/Colab Notebooks/ColabProgetto/Progetto/Modelli/'+nome_dataset+nome_modello+'.h5')

