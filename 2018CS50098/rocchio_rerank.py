import re, sys, csv, json, math
import numpy as np
from stemmer import PorterStemmer
from bs4 import BeautifulSoup

def tokenized(text):
    ps = PorterStemmer()
    tokens = re.split('[.,:;\\\\/\n\t\s\r\'\"\(\)\[\]\{\}]', text)
    tokens = filter(lambda x : True if len(x) > 0 else False, tokens)
    tokens = [ps.stem(token, 0, len(token)-1).lower() for token in tokens]
    return tokens

def rocchiorerank():
    # loading all queries
    infile = open(sys.argv[1],"r")
    topics = BeautifulSoup(infile.read(), 'lxml')
    infile.close()
    querytext = [topic.find('query').text.strip() for topic in topics.find_all('topic')]
    allqueries = [{} for x in querytext]
    for i in range(len(querytext)):
        for token in tokenized(querytext[i]):
            if token not in allqueries[i]:
                allqueries[i][token] = 0
            allqueries[i][token] += 1
    # loading relevant list
    infile = open(sys.argv[2],"r")
    top100 = infile.read().strip().split('\n')
    infile.close()
    allreldocs = set()
    reldocs = [[] for x in allqueries]
    for row in top100:
        col =  row.strip().split()
        reldocs[int(col[0])-1].append(col[2])
        allreldocs.add(col[2])
    # loading all documents
    infile = open(sys.argv[3]+"/metadata.csv","r")
    metadata = list(csv.DictReader(infile))
    infile.close()
    alldocs = {}
    vocabulary = {}
    limit = 2000
    for row in metadata:
        docid = row['cord_uid']
        if docid not in allreldocs:
            if limit > 0:
                limit -= 1
            else:
                continue
        alldocs[docid] = {}
        for header in ['title', 'abstract', 'authors']:
            for token in tokenized(row[header]):
                if token not in alldocs[docid]:
                    alldocs[docid][token] = 0
                alldocs[docid][token] += 1
        pmcpaths = row['pmc_json_files'].split('; ') if row['pmc_json_files'] else []
        pdfpaths = row['pdf_json_files'].split('; ') if row['pdf_json_files'] else []
        paths = pmcpaths if pmcpaths else pdfpaths
        for path in paths:
            with open(sys.argv[3]+"/"+path) as infile:
                doc = json.load(infile)
                for paragraph in doc['body_text']:
                    text = paragraph['text']
                    for token in tokenized(text):
                        if token not in alldocs[docid]:
                            alldocs[docid][token] = 0
                        alldocs[docid][token] += 1
        for token in alldocs[docid]:
            if token not in vocabulary:
                vocabulary[token] = 0
            vocabulary[token] += 1
    # computing tf * idf
    docvector = dict(zip(alldocs.keys(),[[] for x in alldocs]))
    queryvector = [[] for x in allqueries]
    for token in vocabulary:
        idf = math.log(1+(len(alldocs)/vocabulary[token]),2)
        for docid in alldocs:
            tf = (math.log(alldocs[docid][token],2) + 1)/math.log(sum(alldocs[docid].values()),2)
            docvector[docid].append(tf*idf)
        for i in range(len(allqueries)):
            tf = (math.log(allqueries[i][token],2) + 1)/math.log(sum(allqueries[i].values()),2)
            queryvector[i].append()
    # ranking relevant documents
    outfile = open(sys.argv[4],"w")
    for i in range(len(allqueries)):
        dr = np.array([0]*len(docvector.values()[0]))
        dn = np.array([0]*len(docvector.values()[0]))
        for docid in docvector:
            if docid in reldocs[i]:
                dr += docvector[docid]
            else:
                dn += docvector[docid]
        dr = dr/len(reldocs[i])
        dn = dn/(len(docvector)-len(reldocs[i]))
        qm = queryvector[i] + dr + dn
        scores = []
        for docid in reldocs[i]:
            dv = np.array(docvector[docid])
            score = sum(qm * dv)/(math.sqrt(sum(qm * qm)) * math.sqrt(sum(dv * dv)))
            scores.append([score, docid])
        ranks = sorted(scores, reverse=True)
        for j in range(len(ranks)):
            outfile.write(f"{i} Q0 {ranks[j][1]} {j} {ranks[j][0]} runid{1}\n")
    outfile.close()

rocchiorerank()
