import time
start_time = time.time()
import os
import operator
from math import log10, sqrt
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

mytokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()
sortedstopwords = sorted(stopwords.words('english'))
dfs = {}
idfs = {}
speeches = {}
speechvecs = {}
total_word_counts = {}

def tokenize(doc):
    tokens = mytokenizer.tokenize(doc)
    lowertokens = [token.lower() for token in tokens]
    filteredtokens = [stemmer.stem(token) for token in lowertokens if not token in sortedstopwords]
    return filteredtokens

def incdfs(tfvec):
    for token in set(tfvec):
        if token not in dfs:
            dfs[token]=1
            total_word_counts[token] = tfvec[token]
        else:
            dfs[token] += 1
            total_word_counts[token] += tfvec[token]
            

def getcount(token):
    if token in total_word_counts:
        return total_word_counts[token]
    else:
        return 0

def readfiles(corpus_root):
    for filename in os.listdir(corpus_root):
        f = open(os.path.join(corpus_root, filename), "r", encoding='UTF-8')
        doc = f.read()
        f.close() 
        #doc = doc.lower()  
        tokens = tokenize(doc)
        tfvec = Counter(tokens)     
        speeches[filename] = tfvec
        incdfs(tfvec)
    
    ndoc = len(speeches)
    for token,df in dfs.items():
        idfs[token] = log10(ndoc/df)

def calctfidfvec(tfvec, withidf):
    tfidfvec = {}
    veclen = 0.0

    for token in tfvec:
        if withidf:
            tfidf = (1+log10(tfvec[token])) * getidf(token)
        else:
            tfidf = (1+log10(tfvec[token]))
        tfidfvec[token] = tfidf 
        veclen += pow(tfidf,2)

    if veclen > 0:
        for token in tfvec: 
            tfidfvec[token] /= sqrt(veclen)
    
    return tfidfvec
   
def cosinesim(vec1, vec2):
    commonterms = set(vec1).intersection(vec2)
    sim = 0.0
    for token in commonterms:
        sim += vec1[token]*vec2[token]
        
    return sim

def getqvec(qstring):
    tokens = tokenize(qstring)
    tfvec = Counter(tokens)
    qvec = calctfidfvec(tfvec, False)
    return qvec
    
def query(qstring):
    qvec = getqvec(qstring.lower())
    scores = {filename:cosinesim(qvec,tfidfvec) for filename, tfidfvec in speechvecs.items()}  
    return max(scores.items(), key=operator.itemgetter(1))[0]
    
def gettfidfvec(filename):
    return speechvecs[filename]
    
def getidf(token):
    if token not in idfs: 
        return 0
    else: 
        return idfs[token]
    
def docdocsim(filename1,filename2):
    return cosinesim(gettfidfvec(filename1),gettfidfvec(filename2))
    
def querydocsim(qstring,filename):
    return cosinesim(getqvec(qstring),gettfidfvec(filename))

readfiles('C:/Users/NabilPatel/Desktop/DataMining/Assignment-1/presidential_debates')
for filename, tfvec in speeches.items():
    speechvecs[filename] = calctfidfvec(tfvec, True)
	
if __name__ == "__main__":
    print(query("health insurance wall street"))
    print(getcount('health'))
    print("%.12f" % getidf("health"))
    print("%.12f" % docdocsim("1960-09-26.txt", "1980-09-21.txt"))
    print("%.12f" % querydocsim("health insurance wall street", "1996-10-06.txt"))
    print("\n\n")

    print(query("security conference ambassador"))
    print(getcount('attack'))
    print("%.12f" % getidf("agenda"))
    print("%.12f" % docdocsim("1960-10-21.txt", "1980-09-21.txt"))
    print("%.12f" % querydocsim("particular constitutional amendment", "2000-10-03.txt"))

    print("\n\n")

    print(query("particular constitutional amendment"))
    print(getcount('amend'))
    print("%.12f" % getidf("particular"))
    print("%.12f" % docdocsim("1960-09-26.txt", "1960-10-21.txt"))
    print("%.12f" % querydocsim("health insurance wall street", "2000-10-03.txt"))
    print("--- %s seconds ---" % (time.time() - start_time))
