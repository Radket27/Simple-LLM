import numpy as np
import tensorflow as tf
import json

def sample(predictions, temperature):
    predictions = np.array(predictions[0]).astype('float64')
    predictions = np.log(predictions + 1e-7) / temperature
    predictions = np.exp(predictions)
    predictions = predictions / np.sum(predictions)
    return np.random.choice(len(predictions), p=predictions)

def tokenizer_predict(data1,database,max_len):
    """
    Tokenizer for prediction
    """
    tmp_list = []
    a = data1[0].split(' ')
    for x in range(len(a)):
        if(a[x] in database):
            tmp_list.append(database[a[x]])
    if(len(tmp_list) != max_len):
        while(len(tmp_list) != max_len):
            tmp_list.append(0)
    x = np.array(tmp_list)
    return x

def output(data1,database,index,max_len,model,num_words,temperature):
    """
    Output of predicted text
    """
    text = data1[0]
    for i in range(num_words):
        x_predict = tokenizer_predict([text],database,max_len)
        x_predict = x_predict.reshape(1, -1)
        predicted = (model.predict(x_predict))
        predicted_word = str(sample(predicted,temperature))
        if (predicted_word in index):
            predicted_word2 = index[predicted_word]
            text = text + ' ' + predicted_word2
    return text

def main():
    with open("database.json", "r") as f:
        database = json.load(f)
    with open("maxlen", "r") as f:
        max_len = int(f.read())
    with open("index.json", "r") as f:
        index = json.load(f)
    model = tf.keras.models.load_model("model.keras")
    num_words = int(input("How many words?: "))
    temperature = float(input("Temperature: "))
    data_tmp = input("Text: ")
    data1 = []
    data1.append(data_tmp)
    text = output(data1,database,index,max_len,model,num_words,temperature)
    print(text)
    return 0

if(__name__=="__main__"):
    main()