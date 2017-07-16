import os
import sys
import math
import re
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

proc_train_path = "imdb_tr.csv"

# Course paths for training and testv sets 
#train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
#test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation
#stopwords_path = "stopwords.en.txt" # stopwords

#Local Paths
train_path = "D:/JSANTOS/DEVELOPMENT/Data/nlp/aclImdb/train" # use terminal to ls files under this directory
test_path = "D:/JSANTOS/DEVELOPMENT/Data/nlp/aclImdb/predict/imdb_te.csv" # test data for grade evaluation
stopwords_path = "data/stopwords.en.txt" # stopwords


def get_stopwords(path):
    ''' Create an array with the stops words to skip in the documents read
    '''
    stopwords = []
    with open(stopwords_path) as file:
        stopwords = file.readlines()
    # Remove the character \n from the words
    stopwords = list(map(lambda item: item.replace("\n",""), stopwords))
    return [word.lower() for word in stopwords]

# Get the stopwords
stopwords = get_stopwords(stopwords_path)

def clean_text(text):
    ''' Clean the specified text removing the stop words and
    using additional replace for special symbols.
    '''
    # Only consider Alphabet letters , omit numbers or symbols
    text = re.sub('[^a-zA-Z]', ' ', text)
    # REturn and remove Stopwords
    return  ' '.join([word.lower() for word in text.split(" ") 
                           if not word.lower() in stopwords 
                           and word != ''])
       
def get_imdb_filenames(directory, sort =True):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    #Get the file names
    file_names = [filename for root, directories, files in os.walk(directory) for filename in files]
    if sort:
        # Return and ordered list
        return sorted(file_names, key= lambda item: int(item.split('_')[0]))
    # Return default list
    return file_names

def imdb_data_preprocess(inpath, outpath):
    '''Implement this module to extract
    and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have two 
    columns, "text" and label'''
    # Index for Rwo Count
    index = 0
    file_classes = ["pos","neg"]
    classes = [1,0]
    # Creat the output file 
    with open(outpath,'w', encoding="utf8") as output_file:
        # Write the header ,text,polarity
        output_file.write(",text,polarity\n")
        # Loop over the different classes
        for i, file_class in enumerate(file_classes):
            # Read all the files from the inputh/pos directory
            files_path = inpath + "/" + file_class
            files = get_imdb_filenames(files_path,False)
            # Read the file frome the pos_files and using pos_path
            for filename in files:
                with open(files_path + "/" + filename, 
                          encoding="utf8") as input_file:
                    # Get the text from the current file
                    text =  input_file.readline()
                    #Simple preprocessiing for coma value csv
                    text = clean_text(text)
                    # Replace repeated tag
                    text = text.replace('<br />','')
                    # Write in the Output file
                    output_file.write('{},"{}",{}\n'.format(index, 
                                                    text, classes[i]))
                    index += 1

def get_bow_train(inpath, ngrams=1, use_tfidf=False):
    ''' This funcion uses the input file from imdb_data_preprocess
    and create a bag of words using the ngrams specified.The function
    will return and data frame with the columns as teh vocabulary
    in for each row the vectors that correspondos with each document.
    The vector will be the number of ocurrences in the doc no the
    sparse numbering way (0,1)
    '''
    # Load the file using Pandas
    df = pd.read_csv(inpath,index_col=0)
    # Split the text column with the ngrams and find the occurrences
    if use_tfidf:
        count = TfidfVectorizer(ngram_range=(ngrams,ngrams))
    else:
        count = CountVectorizer(ngram_range=(ngrams,ngrams))
    # Extract the words and get the vectors
    bag = count.fit_transform(df["text"])
    # return the current dataframe with the extrated ngrams
    return bag.toarray(), df["polarity"].astype(int).values, count.vocabulary_

def get_bow_test(inpath, ngrams=1, vocabulary=None, use_tfidf=False):
    ''' This funcion uses the input file from imdb_data_preprocess
    and create a bag of words using the ngrams specified.The function
    will return and data frame with the columns as teh vocabulary
    in for each row the vectors that correspondos with each document.
    The vector will be the number of ocurrences in the doc no the
    sparse numbering way (0,1)
    '''
    # Load the file using Pands
    df = pd.read_csv(inpath,index_col=0, encoding = "ISO-8859-1")
    df["text"] = df["text"].map(clean_text)
    # Split the text column with the ngrams and find the occurrences
    if use_tfidf:
        count = TfidfVectorizer(vocabulary=vocabulary,ngram_range=(ngrams,ngrams))
    else:
        count = CountVectorizer(vocabulary=vocabulary,ngram_range=(ngrams,ngrams))
    # Extract the words and get the vectors
    bag = count.fit_transform(df["text"])
    # return the current dataframe with the extrated ngrams
    return bag.toarray()
    
def predict_sentiment_analyis(inpath, testpath, ngrams=1, use_tfidf=False):
    ''' get the Sentiment analysis classification
    '''
     # Get the dataframe with the Bag of words using the ngrams
    train, labels, vocabulary = get_bow_train(inpath, ngrams, use_tfidf)
    # Get the dataframe with the Bag of words using the ngrams
    test = get_bow_test(testpath, ngrams, vocabulary, use_tfidf)
    # Train the NLP model with the training data
    model = linear_model.SGDClassifier(loss='hinge', penalty='l1')
    #Use cross validation to train the model
    cross_val_score(model, train, labels, cv=5)
    # Return the prediction result for the classification
    return model.predict(test)

if __name__ == "__main__":
    
    # Preprocess de data from imdb to generate "imdb_tr.csv"
    #imdb_data_preprocess(train_path, proc_train_path)

    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''

    # # Get the result from the set
    # result = predict_sentiment_analyis(proc_train_path, test_path, 1)
    # # Creat the output file 
    # with open("unigram.output.txt",'w') as output_file:
    #     output_file.write("\n".join([str(value) for value in result]))

    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    bigram.output.txt'''

     # Get the result from the set
    result = predict_sentiment_analyis(proc_train_path, test_path, 2)
    # Creat the output file 
    with open("bigram.output.txt",'w') as output_file:
        output_file.write("\n".join([str(value) for value in result]))

    '''train a SGD classifier using unigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write 
    output to unigramtfidf.output.txt'''

    # Get the result from the set
    result = predict_sentiment_analyis(proc_train_path, test_path, 1, True)
    # Creat the output file 
    with open("unigramtfidf.output.txt",'w') as output_file:
        output_file.write("\n".join([str(value) for value in result]))

    '''train a SGD classifier using bigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write 
    output to bigramtfidf.output.txt'''

    # Get the result from the set
    result = predict_sentiment_analyis(proc_train_path, test_path, 2, True)
    # Creat the output file 
    with open("bigramtfidf.output.txt",'w') as output_file:
        output_file.write("\n".join([str(value) for value in result]))