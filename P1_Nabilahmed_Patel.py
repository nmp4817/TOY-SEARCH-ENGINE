import time
import math
import os
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

fname = []
optitoken = []
tokens_count = []
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
abc = sorted(stopwords.words('english'))
stemmer = PorterStemmer()


def filetolist():
    corpus_root = 'C:/Users/NabilPatel/Desktop/DataMining/Assignment-1/presidential_debates'
    tokencpy1 = []
    tokens = []
    print('in filetolist')
    for filename in os.listdir(corpus_root):
        doc = ''
        token = []
        fname.append(filename)
        file = open(os.path.join(corpus_root, filename), "r", encoding='UTF-8')
        doc = file.read().lower()
        token = tokenizer.tokenize(doc)

        tokenscpy = []
           
        for i in range(len(token)):
            lower_bound = 0
            upper_bound = len(abc)-1
            found = False
            while lower_bound <= upper_bound and not found:
                middle_pos = (lower_bound+upper_bound) // 2
                if abc[middle_pos] < token[i]:
                    lower_bound = middle_pos + 1
                elif abc[middle_pos] > token[i]:
                    upper_bound = middle_pos - 1
                else:
                    found = True
            if found != True:
                exchng = stemmer.stem(token[i])
                token[i] = exchng
                tokenscpy.append(token[i])       

        token = tokenscpy
        tokencpy1 = list(set(token))
        optitoken.append(tokencpy1)
        tokens.append(token)

        token_count = {}
        print('counting...')
        for j in range(len(tokencpy1)):					#tokencpy1 = optitoken[i]
            count = 0
            for k in range(len(token)):
                if token[k] == tokencpy1[j]:
                    count = count + 1
            token_count[tokencpy1[j]] = count
        tokens_count.append(token_count)
    print('after counting')
        
    file.close() 
    
    return tokens
    
def preprocessing(doc):    
    tokens = tokenizer.tokenize(doc)
   
    tokenscpy = []
           
    for i in range(len(tokens)):
        lower_bound = 0
        upper_bound = len(abc)-1
        found = False
        while lower_bound <= upper_bound and not found:
            middle_pos = (lower_bound+upper_bound) // 2
            if abc[middle_pos] < tokens[i]:
               lower_bound = middle_pos + 1
            elif abc[middle_pos] > tokens[i]:
               upper_bound = middle_pos - 1
            else:
                found = True
        if found != True:
            exchng = stemmer.stem(tokens[i])
            tokens[i] = exchng
            tokenscpy.append(tokens[i])       

    tokens = tokenscpy        
   
    return tokens
    
def getcount(strg):	
    print('getcount')
    count = 0
    stemmer1 = PorterStemmer()
    str1 = stemmer1.stem(strg)
    for i in range(len(tokens_count)):
        for k,v in tokens_count[i].items():
            if k == str1:
                count = count + v
                break

    return count
    
    
def query_vector(qstring):
    print('Query Vector')
    qtoken = preprocessing(qstring)
    tf = []
    main_score = {}
    s = 0
    score = 0
    for l in range(len(qtoken)):
        count = countinquery(qtoken[l],qtoken)
        
        if count != 0:
            score = (1 + math.log10(count))
        else:
            score = 0
        tf.append(score)
        s = s + score*score     
        
    for l in range(len(qtoken)):
        score = tf[l]/math.sqrt(s)   
        main_score[qtoken[l]] = score
    
    return main_score

def countinquery(strg, doc):
    count = 0
    for i in range(len(doc)):
            if doc[i] == strg:
                count = count + 1
    return count


def getidf(strg):
    df_t = 0
    for i in range(len(optitoken)):
        for j in range(len(optitoken[i])):
            if optitoken[i][j] == strg:
                df_t = df_t + 1
                break
    if df_t != 0:
        return math.log10(30/df_t)
    else:
        return 0

def countindoc(strg, doc_count):
    for k,v in doc_count.items():
            if k == strg:
                return v
    
def get_tfidf(otoken,l):
    print('Get TF-IDF')

    tfidf_doc = {}
    print('counting...')
    tfidf = 0
    s = 0

    for i in range(len(otoken)):
        idf = getidf(otoken[i])
        tf = countindoc(otoken[i],tokens_count[l])
        tfidf =  (1 + math.log10(tf))*(idf)
        s = s + (tfidf*tfidf)
        tfidf_doc[otoken[i]] = tfidf
        
    for k,v in tfidf_doc.items():
        tfidf_doc[k]= v/math.sqrt(s)
            
    return tfidf_doc    

def query_alldoc_sim(qvector, tfidf):
    print('Query-all_Doc Similarity')

    s = 0
    for qk, qv in qvector.items():
        product = 0
        for tk, tv in tfidf.items():
            if qk == tk:
                product = qv * tv
                break
        s = s + product
   
    return s

def query1(qstring):
    print('Query')
    tfidf = []
    tfidf_doc = {}
    cosine = []
    qvector = query_vector(qstring)
    for l in range(len(optitoken)):
        tfidf_doc = get_tfidf(optitoken[l],l)
        tfidf.append(tfidf_doc)
        cos = query_alldoc_sim(qvector, tfidf[l])
        cosine.append(cos)

    return fname[cosine.index(max(cosine))]

def querydocsim(qstring,doc):
    print('Query_doc_sim')
    qvector = query_vector(qstring)
    l = fname.index(doc)
    tfidf_doc = get_tfidf(optitoken[l],l)
    cos = query_alldoc_sim(qvector,tfidf_doc)

    return cos

def docdocsim(doc1,doc2):
    print('Doc_doc_sim')
    l = fname.index(doc1) 
    tfidf_doc1 = get_tfidf(optitoken[l],l)
    l = fname.index(doc2)
    tfidf_doc2 = get_tfidf(optitoken[l],l)
    cos = query_alldoc_sim(tfidf_doc1,tfidf_doc2)

    return cos    
    
def main():

    start_time = time.time()

    filetolist()
    
    print("--- %s seconds ---" % (time.time() - start_time))

main()