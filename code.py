# Python 3.8
import requests
import os
import pandas as pd
import glob
from re import sub
import numpy as np
from gensim.utils import simple_preprocess
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
from gensim.similarities import WordEmbeddingSimilarityIndex
from nltk.corpus import stopwords
import nltk

from bs4 import BeautifulSoup

df = pd.read_excel("keywords.xlsx", index_col=0)

BBC_MAX_PAGES = 29
BBC_ARTICLES_PP = 10
ARTICLES_TO_FETCH = 100


# Problem 1&2 - Fetches the top 100 (or less) articles associated with each keyword given into folder, then parses
# their content and saves it to "keyword/article_id.txt"
def remove_spaces(w, rep="_"):
    return w.replace(" ", rep)


def process_results(w, txt, mx=BBC_ARTICLES_PP):
    try:
        os.makedirs(w)
    except OSError:
        pass

    soup = BeautifulSoup(txt, 'html.parser')
    ul = soup.main.find("ul")
    
    extracted = 0
    for i in ul.children:
        a = i.find("a")["href"]
        # Works to exlcude any non-article content via only allowing links containin gnews and disallowing:
        #   -> learning, programmes or bbc/today (video content)

        # TODO verify this is good enough for determining article relevance
        # prev: not("news" in a) or ("learning" in a) or ("programmes" in a) or ("/today/" in a):
        if (a is None) or (extracted >= mx) or ("programmes" in a) or not("technology" in a):
            continue
        

        r = requests.get(a)
        extracted += extract_content(w, a, r.text)
    
    return extracted


# Problem 2 - Processes articles producing "keyword/article_id.txt" files containing the article content
def get_article_content(soup):
    article = soup.find("article")
    headlinestory = soup.find("div", {"class": "headlinestory"})
    storybody = soup.find("td", {"class": "storybody"})
    newsroundnav = soup.find("div", {"class": "newsround-cbbc-nav-wrapper"})
    
    # Each query selector here has been tested on it's relevant format of pages to grab the heading of the article,
    # content and subheadings in the order it appears on the page
    if article is not None:
        # article elem only exists on modern BBC news pages
        return article.select("#main-heading, div[data-component='text-block'], div[data-component='crosshead-block']")
    elif headlinestory is not None:
        # .topStoryH only exists on really old 'BBC news online' pages
        return headlinestory.parent.select("b, p")
    elif storybody is not None:
        # .storybody only exists on older 'BBC news magazine' pages
        return soup.select("td>div>h1, .storybody>p")
    elif newsroundnav is not None:
        return soup.select("h1.newsround-story-header__title-text, p.newsround-story-body__text")
    
    # Fallback which works for most old(ish) BBC news/sport/election etc. articles which follow the same(ish) format
    fallback = soup.select(".sh, font>b, font>p")
    if len(fallback) == 0:
        return None
    
    return fallback       


def extract_content(w, link, txt):
    soup = BeautifulSoup(txt, 'html5lib')
    content = get_article_content(soup)

    if content is None:
        print("    -> Failed to parse story (bad format): %s" % link)
        return 0
    
    path = "%s/%s.txt" % (w, link.replace("/", "").replace("https", "").replace("http", "").replace(":", ""))

    new_file = open(path, "w", encoding="utf-8")

    for i in content:
        t = i.text.strip()
        if t == "":
            continue
        new_file.write(t + "\n")

    new_file.close()
    print("    -> Parsed story: %s" % link)

    return 1


def problem1_2(words):
    for w in words:
        print("Processing: %s" % w)
        fetched = 0
        page = 1
        while fetched < ARTICLES_TO_FETCH and page <= BBC_MAX_PAGES:
            url = "https://www.bbc.co.uk/search?q=" + remove_spaces(w, "+") + "&page=" + str(page)
            r = requests.get(url)
            fetched += process_results(remove_spaces(w), r.text, min(ARTICLES_TO_FETCH - fetched, BBC_MAX_PAGES))
            page += 1
        print("Extracted content from %s of the articles for %s.\n" % (str(fetched), w))

def semantic_distance(word, docs, stop_words, similarity_index, load=True, save=True):
    def preprocess(doc):
        return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stop_words]
    
    query = preprocess(word)
    documents = []

    # Read in docs
    for i in glob.glob("%s/*.txt" % remove_spaces(docs)):
        with open(i, 'r', encoding='utf-8') as f:
            documents.append(f.read())

    corpus = [preprocess(document) for document in documents]

    # Build the term dictionary, TF-idf model
    dictionary = Dictionary(corpus + [query])
    tfidf = TfidfModel(dictionary=dictionary)

    path = "sstm/%s-on-%s.sstm" % (remove_spaces(word), remove_spaces(docs))
    if os.path.isfile(path) and load:
        print("    -> Found saved sstm!")
        similarity_matrix = SparseTermSimilarityMatrix.load(path)
    else:
        if load:
            print("    -> No saved sstm found!")
        print("    -> Generating sstm now (this may take a while!)...")

        # Create the term similarity matrix. SLOW!
        similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

        if save:
            print("    -> Saving generated sstm!")
            similarity_matrix.save(path)
    
    # Compute Soft Cosine Measure between the query and the documents.
    # From: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb
    query_tf = tfidf[dictionary.doc2bow(query)]

    index = SoftCosineSimilarity(
                tfidf[[dictionary.doc2bow(document) for document in corpus]],
                similarity_matrix)

    doc_similarity_scores = index[query_tf]
    return doc_similarity_scores.mean()


def problem3(words):
    print("Processing semantic distances, first loading some utils!\n")
    # Download stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english')) 

    # Load the model: this is a big file, can take a while to download and open
    glove = api.load("glove-wiki-gigaword-50")    
    similarity_index = WordEmbeddingSimilarityIndex(glove)

    print("")

    for w1 in words:
        for w2 in words:
            print("Processing semantic distance of %s onto corpus of %s" % (w1, w2))
            sd = semantic_distance(w1, w2, stop_words, similarity_index)
            print("Semantic distance of %s onto corpus of %s = %.3f\n" % (w1, w2, sd))

            # therefore df[w1 = col][w2 = row] = semantic distance of w1 on docs for w2
            df[w1][w2] = sd
    
    print("Saved calculated semantic distances to 'distances.xlsx'!")
    df.to_excel("distances.xlsx")
    

#problem1_2(df.columns)
#problem3(df.columns)