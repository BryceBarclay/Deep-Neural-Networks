import numpy as np 
import pandas as pd 

# part (a) does not yet print out the number of unique words

# (a)

# load text
with open('Plato_Republic.txt', 'r') as myfile:
    myText = myfile.read().replace('\n',' ').lower() 
#print(myText[0:40]) # check that format is lowercase text

# tokenize text
import nltk 
# nltk.download('punkt')
nltk.data.path.append("/DNN_HW3/nltk_data") 
myText_tokenized = nltk.word_tokenize(myText) 
# 
print(myText_tokenized[0:10]) # check if properly tokenized
print("The total number of words N in the text : ",len(myText_tokenized))
myUnigram = nltk.ngrams(myText_tokenized, 1) 
fdist_uni = nltk.FreqDist(myUnigram) 
print("The total number of unique words in the text : ",len(fdist_uni))

# (b)

myUnigram = nltk.ngrams(myText_tokenized, 1) 
fdist_uni = nltk.FreqDist(myUnigram) 
#print(len(fdist_uni.most_common()[0][0][0]))
com_words8 = []
for i in range(0,len(myText_tokenized)):
    if(len(fdist_uni.most_common()[i][0][0]) > 7):
        com_words8.append(fdist_uni.most_common()[i][0][0]) # tuple in a tuple 
    if(len(com_words8) == 5):
        break
print("The 5 most common words with at least 8 characters:") 
print(com_words8)


# (c)

myBigram = nltk.ngrams(myText_tokenized, 2) 
fdist_bi = nltk.FreqDist(myBigram) 
#print("The 10 most frequent coupled words:\n") # check that bigram works properly
#print(fdist_bi.most_common()[0:10])

def prob(x_1,x_2):
    return fdist_bi[(x_1,x_2)]/fdist_uni[(x_1,)]

#print(prob('and','the')) # check that function works
#print(prob(',','and'))


# (d)

# Note that perplexity is calculated without the zero probabliity events
N = len(myText_tokenized)
PP = 0
for i in range(0,N-1):
    #if(prob(myText_tokenized[i],myText_tokenized[i+1]) > 0):
    PP = PP + np.log(prob(myText_tokenized[i],myText_tokenized[i+1]))
PP = PP*(-1/(N-1))
PP = np.exp(PP)

print('The perplexity is: ', PP)

