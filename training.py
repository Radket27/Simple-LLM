import numpy as np
import tensorflow as tf
import json
from sys import argv

def file_transport():
    name = []
    if(len(argv) == 1):
        print("Set arguments")
        exit()
    for i in range(len(argv)):
        if(i == 0):
            pass
        else:
            name.append(argv[i])
    return name[0]

def tokenizer(data,database):
    """
    Tokenizer
    """
    tmp_list2 = []
    max_len = 0
    x_tmp = []
    y_tmp = []
    for i in data:
        a = i.split(' ')
        token = []
        for x1 in a:
            if(x1 in database):
                token.append(database[x1])
        if(len(token) > 1):
            tmp_list2.append(token)
    for x1 in tmp_list2:
        if len(x1) > max_len:
            max_len = len(x1)

    for x1 in tmp_list2:
        pad = x1 + [0] * (max_len - len(x1))
        x_tmp.append(pad[:-1])
        y_tmp.append(pad[-1])
    
    x = np.array(x_tmp)
    y = np.array(y_tmp)
    max_len = max_len - 1
    return x,y,max_len

def database_generator(data):
    """
    A dictionary for each word and number associated with it
    """
    database = {}
    index = {}
    next_number = 1
    for i in data:
        a = i.split(' ')
        for x in range(len(a)):
            if(not(a[x] in database)):
                database[a[x]] = next_number
                index[next_number] = a[x]
                next_number += 1
    return database, index

def model_llm(x,y,num_class):
    """
    Keras model
    """
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim = num_class, output_dim=64, input_length = x.shape[1]))
    model.add(tf.keras.layers.LSTM(128, dropout=0.5, recurrent_dropout=0.3))
    model.add(tf.keras.layers.Dense(num_class,activation='softmax'))
    model.compile(optimizer='Adam', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                  metrics=['accuracy'])
    model.summary()
    model.fit(x,y, epochs=50, validation_split=0.1, batch_size=64, callbacks=[callback])
    return model


def main():
    data = []
    with open(str(file_transport()),'r') as f:
        for i in f:
            if(i.strip()):
                data.append(i.rstrip("\n"))
    database, index = database_generator(data)
    x,y,max_len = tokenizer(data,database)
    num_class = (int(len(database))) + 1
    model = model_llm(x,y,num_class)
    model.save("model.keras")
    with open("maxlen", "w") as f:
        f.write(str(max_len))
    with open("database.json", "w") as f:
        json.dump(database,f, ensure_ascii=False)
    with open("index.json","w") as f:
        json.dump(index, f, ensure_ascii=False)
    return 0


if(__name__=="__main__"):
    main()
