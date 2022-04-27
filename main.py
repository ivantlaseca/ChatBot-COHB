import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data2.pickle", "rb") as f:  # Delete old pickle file & model if new intents are added
        words, labels, training, output = pickle.load(f)  # Save 4 variables into pickle file. If saved & works fine, load in the lists.
except:
    words = []
    labels = []
    docs_x = []  # List of all patterns
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)  # Tokenizes each pattern in all of the patterns. Ex: "Whats up?" -> "What"
            words.extend(wrds)  # Adds all of the wrds to the words list
            docs_x.append(wrds)  # Adds all of the patterns to the docs list
            docs_y.append(intent["tag"])  # Each entry in docs_x corresponds to an entry in docs_y

        if intent["tag"] not in labels:
            labels.append(intent["tag"])  # Adds all tags to labels list.

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]  # Stemmer reduces words to their root words
    words = sorted(list(set(words)))  # Set removes duplicate elements. List converts it back into a list.
    labels = sorted(labels)

    # Training

    training = []  # Bags of words(lists of 0s and 1s)
    output = []  # List of 0s and 1s

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:  # If the word exists, place a 1
                bag.append(1)
            else:
                bag.append(0)

        # Generating Output
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1  # Find indx of the tag in the labels list and set that value to 1 in output_row

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data2.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)  # "Write these variables into a pickle file"

# Start part 3

tensorflow.compat.v1.reset_default_graph()          #Potential issue

# Take in a bag of words (sentence) and output what we think we should respond, based on the probabilities for each tag.
# Ideally, when training we want these hidden layers to figure out/learn what words represent which output.
# Basically classifying sentences to tags.

net = tflearn.input_data(shape=[None, len(training[0])])  # Defines input shape that we're expecting for our model
net = tflearn.fully_connected(net, 8)  # 8 Neurons for our first hidden layer
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  # Allows us to get probabilities for each neuron
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    # Adjust the model to adjust the accuracy. Adding more tags to our intents will likely drop the accuracy slightly.
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)  # Pass our training data to the model. n_epoch is the # of times it will see the same data (training sets). Mess with this # w/ more or less.
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:                 #Word exists
                bag[i] = 1
    return numpy.array(bag)

#Ask user for some kind of sentence and spit out a response.
def chat():

    intnts = data["intents"]
    greetings = intnts[0]
    greeting = random.choice(greetings['responses'])
    print(greeting)
    while True:
            inp = input("You: ")
            if inp.lower() == "quit":
                break

            results = model.predict([bag_of_words(inp,words)])[0]
            results_index = numpy.argmax(results)                   #Index of the greatest value in our results list
            tag = labels[results_index]

            if results[results_index] > 0.7:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg["responses"]
                print(random.choice(responses))
            else:
                print("I didn't get that. Try again.")

chat()