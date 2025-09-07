import numpy as np
import tensorflow as tf
import json

def file_transport():
    from sys import argv
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
    y_tmp = []
    for i in data:
        tmp_list = []
        a = i.split(' ')
        if(max_len < len(a)):
            max_len = len(a)
        for x in range(len(a)):
            tmp_list.append(database[(database.index(a[x]) + 1)])
        tmp_list2.append(tmp_list)
    for i in range(len(tmp_list2)):
        if(len(tmp_list2[i]) != max_len):
            while(len(tmp_list2[i]) != max_len):
                tmp_list2[i].append(-1)
    for i in range(len(tmp_list2)):
        y_tmp.append((tmp_list2[i][-1]))
        del tmp_list2[i][-1]
    x = np.array(tmp_list2)
    y = np.array(y_tmp)
    return x,y,max_len

def database_generator(data):
    """
    A dictionary for each word and number associated with it
    """
    database = []
    next_number = 0
    for i in data:
        a = i.split(' ')
        for x in range(len(a)):
            if(a[x] in database):
                pass
            else:
                database.append(a[x])
                database.append(next_number)
                next_number += 1
    database.append(' ')
    database.append(-1)
    return database

def model_llm(x,y,num_class):
    """
    Keras model
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape = (x.shape[1],)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(num_class,activation='softmax'))
    model.compile(optimizer='Adam', 
                  loss=tf.keras.losses.CategoricalCrossentropy(), 
                  metrics=['accuracy'])
    model.summary()
    model.fit(x,y, epochs=10)
    return model


def main():
    data = []
    with open(str(file_transport()),'r') as f:
        for i in f:
            data.append(i.rstrip("\n"))
    database = database_generator(data)
    x,y,max_len = tokenizer(data,database)
    num_class = (int(len(database)/2))
    y = np.eye(num_class)[y]
    model = model_llm(x,y,num_class)
    model.save("model.keras")
    with open("maxlen", "w") as f:
        f.write(str(max_len))
    with open("database.json", "w") as f:
        json.dump(database,f, ensure_ascii=False)    
    return 0


if(__name__=="__main__"):
    main()
