import numpy as np
import tensorflow as tf
import json

def tokenizer_predict(data1,database,max_len):
    """
    Tokenizer for prediction
    """
    tmp_list = []
    a = data1[0].split(' ')
    for x in range(len(a)):
        tmp_list.append(database[(database.index(a[x]) + 1)])
    if(len(tmp_list) != max_len):
        while(len(tmp_list) != max_len):
            tmp_list.append(-1)
    x = np.array(tmp_list)
    return x

def output(data1,database,max_len,model):
    """
    Output of predicted text
    """
    x_predict = tokenizer_predict(data1,database,max_len-1)
    x_predict = x_predict.reshape(1, -1)
    predicted = (model.predict(x_predict))
    print(database[::2])
    print(predicted)
    generated_database = database[::2]
    print(np.argmax(predicted))
    generated_database[np.argmax(predicted)]
    print(str(data1[0])+' '+str(generated_database[np.argmax(predicted)]))

def main():
    with open("database.json", "r") as f:
        database = json.load(f)
    with open("maxlen", "r") as f:
        max_len = int(f.read())
    model = tf.keras.models.load_model("model.keras")
    data_tmp = input("Text: ")
    data1 = []
    data1.append(data_tmp)
    output(data1,database,max_len,model)
    return 0

if(__name__=="__main__"):
    main()