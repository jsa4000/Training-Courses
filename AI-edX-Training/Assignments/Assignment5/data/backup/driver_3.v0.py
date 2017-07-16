import os
import sys
import math
import numpy as np
import pandas as pd

# Global variables wheter files will be generated or not
GENERATE_FILES = True

# Course paths for training and testv sets 
#train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
#test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation
#stopwords_path = "stopwords.en.txt" # stopwords

#Local Paths
train_path = "D:/JSANTOS/DEVELOPMENT/Data/nlp/aclImdb/train" # use terminal to ls files under this directory
test_path = "D:/JSANTOS/DEVELOPMENT/Data/nlp/aclImdb/Outputs/imdb_te.csv" # test data for grade evaluation
stopwords_path = "data/stopwords.en.txt" # stopwords

def get_stopwords(path):
    ''' Create an array with the stops words to skip in the documents read
    '''
    stop_words = []
    with open(stopwords_path) as file:
        stop_words = file.readlines()
    # Remove the character \n from the words
    return list(map(lambda item: item.replace("\n",""),stop_words))

def clean_text(text, stopwords,remove_special=False):
    ''' Clean the specified text removing the stop words and
    using additional replace for special symbols.
    '''
    # Remove Stopwords
    text = ' '.join([word.lower() for word in text.split(" ") 
                            if not word.lower() in stopwords])
    # Replace repeated tag
    text = text.replace('<br />','')

    # Check to remove Special Characters
    if remove_special:
        # Remove special symbols
        symbols = ("()',.;*-+¿¡!#?:~")
        for symbol in symbols:
            text = text.replace(symbol,'')
        text = text.replace('"','')
        text = text.replace('  ',' ')
    
    #return the final text clean
    return text
    
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

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    '''Implement this module to extract
    and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have two 
    columns, "text" and label'''

    # Get the stopwords
    stopwords = get_stopwords(stopwords_path)
    stopwords = [word.lower() for word in stopwords]

    # Index for Rwo Count
    index = 0
    file_classes = ["pos","neg"]
    classes = [1,0]

    # Creat the output file 
    with open(outpath + name,'w', encoding="utf8") as output_file:
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
                    # Clean the text using the stopwords and addtiona behaviour
                    text = clean_text(text, stopwords)
                    # Write in the Output file
                    output_file.write('{},"{}",{}\n'.format(index, 
                                                    text, classes[i]))
                    index += 1

def get_bag_of_words(input,ngrams=1):
    ''' This funcion uses the input file from imdb_data_preprocess
    and create a bag of words using the ngrams specified.The function
    will return and data frame with the columns as teh vocabulary
    in for each row the vectors that correspondos with each document.
    The vector will be the number of ocurrences in the doc no the
    sparse numbering way (0,1)
    '''
    # Load the file using Pands
    df = pd.read_csv(input,index_col=0)
    # Split the text column with the ngrams and find the occurrences
    bow = df["text"].map(lambda text: dict((word, text.count(word)) 
                                            for word in text.split(" ")
                                            if word != ''))
    # Convert bag of words from map to a dataframe with columns
    df_bow = pd.DataFrame.from_records(bow).fillna(0)
    # return the current dataframe with the extrated ngrams
    return df_bow.astype(int), df["polarity"].astype(int)
    
if __name__ == "__main__":
    
    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''

    # Preprocess de data from imdb to generate "imdb_tr.csv"
    if GENERATE_FILES: imdb_data_preprocess(train_path, name="imdb_tr.csv")

    # # Get the dataframe with the Bag of words using the ngrams
    # train, labels = get_bag_of_words("imdb_tr.csv", 1)
  
    # # Train the NLP model with the training data
    # print(train.head())
    # print(labels.head())

    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    bigram.output.txt'''

    '''train a SGD classifier using unigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write 
    output to unigramtfidf.output.txt'''

    '''train a SGD classifier using bigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write 
    output to bigramtfidf.output.txt'''
    pass