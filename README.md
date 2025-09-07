# Simple-LLM
## How to use
Install TensorFlow and NumPy.
```
pip install -r requirements.txt
```
Edit the Keras model if needed, then train it.
```
nano training.py
```
Train the model on your data.
```
python3 training.py name_of_your_file.txt
```
Finally, generate text using your trained model.
```
python3 prediction.py
```