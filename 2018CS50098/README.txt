How to run the Program?
unzip 2018CS50098.zip
cd 20XXCSXX999
bash build.sh
bash rocchio_rerank.sh [query-file] [top-100-file] [collection-file] [output-file]
bash lm_rerank.sh [rm1|rm2] [query-file] [top-100-file] [collection-dir] [output-file] [expansions-file]

Programming Language used: Python 3.7.11
Libraries used: numpy, beautiful-soup, lxml

Rocchio Implementation:
Approximation used for non-relevent documents: 2000 random choosen documents other than those mentioned in top100
Vocabulary used: All words in top100 and 2000 randomly choosen non-relevent documents
tf used: log(1+ word frequency)/(1+log(1+ total word count))
idf used: log(1+(total number of documents in collection/total number of documents with the term))

