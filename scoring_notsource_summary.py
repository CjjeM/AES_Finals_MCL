import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import language_tool_python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from time import sleep
from threading import Thread
import time
import re
import nltk

lt_servers = []
thread_list = []
print("Initializing Language Tool")
start = time.time()

for i in range(5):
    lt_servers.append(language_tool_python.LanguageTool('en-US'))
    
stop_words = stopwords.words('english')

df = pd.read_excel("training_set_rel3.xls")

# custom thread
class LanguageCheck(Thread):
    # constructor
    def __init__(self, df, idx):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.value = None
        self.df = df
        self.index = idx
 
    # function executed in a new thread
    def run(self):
        self.df['grammar_errors'] = self.df['essay'].apply(self.grammar_errors)
        self.value = self.df
        return
    
    def grammar_errors(self, essay):
        errors = lt_servers[self.index].check(essay)
        return len(errors)

# custom thread
class LanguageCorrect(Thread):
    # constructor
    def __init__(self, df, idx):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.value = None
        self.df = df
        self.index = idx
 
    # function executed in a new thread
    def run(self):
        self.df['essay'] = self.df['essay'].apply(self.autocorrect_essay)
        self.value = self.df
        return
    
    def autocorrect_essay(self, essay):
        corrected_essay = lt_servers[self.index].correct(essay)
        return corrected_essay

# essay structure

def word_count(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)

    return len(words)

def unique_word_count(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)
    unique_words = set(words)

    return len(unique_words)

def sentence_count(essay):
    sentences = nltk.sent_tokenize(essay)
    
    return len(sentences)

def avg_word_len(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)
    
    return sum(len(word) for word in words) / len(words)


def sentence_to_wordlist(raw_sentence):
    
    clean_sentence = re.sub("[^a-zA-Z0-9]"," ", raw_sentence)
    tokens = nltk.word_tokenize(clean_sentence)
    
    return tokens

def tokenize(essay):
    stripped_essay = essay.strip()
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)
    
    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))
    
    return tokenized_sentences

def count_pos(essay):
    
    tokenized_sentences = tokenize(essay)
    
    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0
    
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)
        
        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]
        
            if pos_tag.startswith('N'): 
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1
            
    return noun_count, adj_count, verb_count, adv_count

def preprocess_dataframe(df, essay_set):
    df = df[df["essay_set"]==essay_set]
    print(f"Retrieving Essay Set #{essay_set}")
    print(f"Dataframe shape: {df.shape}")
    clean_df = df[['essay', 'domain1_score']].copy()

    clean_df = clean_df.rename(columns={'domain1_score': 'actual_score'})

    # get essay structure
    print("Getting Word Count")
    clean_df['word_count'] = clean_df['essay'].apply(word_count)
    print("Getting Unique Word Count")
    clean_df['unique_word_count'] = clean_df['essay'].apply(unique_word_count)
    print("Getting Sentence Count")
    clean_df['sentence_count'] = clean_df['essay'].apply(sentence_count)
    print("Getting Average Word Length")
    clean_df['avg_word_len'] = clean_df['essay'].apply(avg_word_len)
    print("POS Tagging")
    clean_df['noun_count'], clean_df['adj_count'], clean_df['verb_count'], clean_df['adv_count'] = zip(*clean_df['essay'].map(count_pos))

    # get grammatical errors
    print("Getting Grammatical Errors")
    df_split = np.array_split(clean_df, len(lt_servers))
    # put threads into list
    for idx, i in enumerate(df_split):
        thread_langcheck = LanguageCheck(df=i, idx=idx)
        thread_list.append(thread_langcheck)

    # start thread list
    for thread in thread_list:
        thread.start()

    # join all threads
    for thread in thread_list:
        thread.join()
    
    clean_df = pd.concat([thread.value for thread in thread_list], axis=0)
    
    thread_list.clear()

    # autocorrect errors
    print("Autocorrecting Essay")
    df_split = np.array_split(clean_df, len(lt_servers))
    # put threads into list
    for idx, i in enumerate(df_split):
        thread_langcheck = LanguageCorrect(df=i, idx=idx)
        thread_list.append(thread_langcheck)

    # start thread list
    for thread in thread_list:
        thread.start()

    # join all threads
    for thread in thread_list:
        thread.join()
    
    clean_df = pd.concat([thread.value for thread in thread_list], axis=0)
    
    thread_list.clear()

    for tool in lt_servers:
        tool.close()

    # preprocess essay for tokenization
    print("Preprocess for tokenization")
    clean_df.reset_index(drop=True, inplace=True)
    clean_df['essay'] = clean_df['essay'].str.replace("[^a-zA-Z#]", " ")
    clean_df['essay'] = clean_df['essay'].apply(lambda x: x.lower())

    # tokenization
    print("Tokenization Start")
    tokenized_doc = clean_df['essay'].apply(lambda x: x.split())

    # remove stop-words
    print("Removing Stop Words")
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

    # stemming
    print("Word Stemming")
    porter_stemmer = PorterStemmer()
    tokenized_doc = tokenized_doc.apply(lambda x: [porter_stemmer.stem(item) for item in x])

    # de-tokenization
    print("Detokenize")
    detokenized_doc = []
    for i in range(len(clean_df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)

    clean_df['essay'] = detokenized_doc

    return clean_df


def scorer_no_lsa_similarity():
    # Essay Set 1, max_features = 10000, min_df = 5
    # Essay Set 7, max_features = 10000, min_df = 5

    essay_set = 7
    max_features = 10000
    min_df = 5
    
    start_preprocess = time.time()

    print("Preprocess Start")
    clean_df = preprocess_dataframe(df, essay_set)

    print("Creating TF-IDF Vectorizer")
    # Create a vectorizer for the training data
    tokenizer = RegexpTokenizer(r'\w+')

    # Vectorize document using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(lowercase=True,
                                            stop_words='english',
                                            ngram_range = (1,3),
                                            tokenizer = tokenizer.tokenize,
                                            max_features=max_features,
                                            max_df=0.8,
                                            min_df=min_df)
    print("Building Matrix")
    tfidf_matrix = tfidf_vectorizer.fit_transform(clean_df["essay"])
    print(f"Train TFIDF Matrix Shape: {tfidf_matrix.shape}")

    print("Convert TF-IDF matrix to SVD")
    # TFIDF to SVD
    svd_model = TruncatedSVD(n_components=100,
                            n_iter=200,
                            random_state=69)
        
    svd = svd_model.fit_transform(tfidf_matrix)
    #normalized_svd = Normalizer(copy=False).fit_transform(svd)

    end_preprocess = time.time()

    print(f"Preprocess time: {end_preprocess - start_preprocess}")

    print("Training Start")

    start_training = time.time()

    print("Getting Features")
    x_df_features = clean_df[['word_count', 
                            'unique_word_count',
                            'sentence_count',
                            'avg_word_len',
                            'grammar_errors', 
                            'noun_count',
                            'adj_count',
                            'verb_count',
                            'adv_count']]
    x_features = np.concatenate((x_df_features.to_numpy(), svd), axis=1)
    y_features = clean_df['actual_score'].to_numpy()

    print("Splitting Dataset")
    x_train, x_test, y_train, y_test = train_test_split(x_features, y_features, test_size = 0.2, train_size = 0.8, random_state = 420)

    print("Building Linear Regression Model")
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)

    print("Building SVR Model")
    svr_model = SVR()
    svr_model.fit(x_train, y_train)

    print("Building Decision Tree Model")
    tree_model = DecisionTreeRegressor()
    tree_model.fit(x_train, y_train)

    print("Building Bayesian Regressor")
    bayes_model = BayesianRidge()
    bayes_model.fit(x_train, y_train)

    print("Building AdaBoost Regressor")
    ada_model = AdaBoostRegressor(n_estimators=100)
    ada_model.fit(x_train, y_train)

    print("Building Random Forest Regressor")
    ran_model = RandomForestRegressor()
    ran_model.fit(x_train, y_train)

    print("Building Gradient Boosting Regressor")
    grad_model = GradientBoostingRegressor(n_estimators=200)
    grad_model.fit(x_train, y_train)

    print("Building Logistic Regression Model")
    log_model = LogisticRegression(solver="saga", max_iter=10000)
    log_model.fit(x_train, y_train)

    print("Getting Predictions")
    predictions = [ lr_model.predict(x_test),
                    svr_model.predict(x_test),
                    tree_model.predict(x_test),
                    bayes_model.predict(x_test),
                    ada_model.predict(x_test),
                    ran_model.predict(x_test),
                    grad_model.predict(x_test),
                    log_model.predict(x_test)]
    scores = []
    
    for idx, pred in enumerate(predictions):
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r_score = r2_score(y_test, pred)

        scores.append([idx, mae, mse, rmse, r_score])
    
    print("\nResults:")
    best_score = max(scores, key=lambda sublist: sublist[-1])
    print(f"Model {best_score[0]}")
    print(f"Mean Absolute Error: {best_score[1]}")
    print(f"Mean Squared Error: {best_score[2]}")
    print(f"Root Mean Squared Error: {best_score[3]}")
    print(f"R2 score: {best_score[4]}\n")

    print("Cross Validation 10-Folds")
    kf = KFold(n_splits=10)

    scores = [cross_val_score(lr_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(svr_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(tree_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(bayes_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(ada_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(ran_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(grad_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(log_model, x_features, y_features, cv=kf).mean()]
    
    print(f"Model {scores.index(max(scores))}")
    print(f"Overall Score: {max(scores)}\n")

    end_training = time.time()

    print(f"Training time: {end_training - start_training}")


def scorer_with_lsa_similarity():
    # Essay Set 1, sample_essays=10
    # Essay Set 7, sample_essays=10

    essay_set = 7
    sample_essays = 10

    start_preprocess = time.time()
    
    print("Preprocess Start")
    clean_df = preprocess_dataframe(df, essay_set)

    end_preprocess = time.time()

    print(f"===Preprocess time: {end_preprocess - start_preprocess}===")

    start_vector = time.time()

    df_lsa = clean_df.copy()
    largest_possible_score = df_lsa.nlargest(1, 'actual_score')['actual_score'].values[0]

    top_score = largest_possible_score - (largest_possible_score * 0.10)

    chosen_essay = df_lsa[df_lsa['actual_score'] >= top_score]
    chosen_essay = chosen_essay.groupby('actual_score').sample(sample_essays, random_state=26)

    df_lsa = df_lsa.drop(index = chosen_essay.index)

    # Create a vectorizer for lsa similarity
    tokenizer = RegexpTokenizer(r'\w+')

    # Vectorize document using TF-IDF
    tfidf_lsa_vectorizer = TfidfVectorizer(lowercase=True,
                                            stop_words='english',
                                            ngram_range = (1,3),
                                            tokenizer = tokenizer.tokenize)

    tfidf_lsa_matrix = tfidf_lsa_vectorizer.fit_transform(chosen_essay["essay"])

    # TFIDF to SVD
    svd_lsa_model = TruncatedSVD(n_components=100,
                            n_iter=200,
                            random_state=69)
        
    svd_lsa = svd_lsa_model.fit_transform(tfidf_lsa_matrix)
    normalized_svd = Normalizer(copy=False).fit_transform(svd_lsa)

    
    def lsa_score(essay):
        essay_matrix = tfidf_lsa_vectorizer.transform([essay])
        essay_svd = svd_lsa_model.transform(essay_matrix)
        normalized_essay_svd = Normalizer(copy=False).fit_transform(essay_svd)

        # Compare current essay to the top 10% scored essay
        similarities = cosine_similarity(normalized_svd, normalized_essay_svd).max()

        return similarities.max()
    
    df_lsa['lsa_score'] = df_lsa['essay'].apply(lsa_score)


    # Create a vectorizer for the training data
    tokenizer = RegexpTokenizer(r'\w+')

    # Vectorize document using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(lowercase=True,
                                    stop_words='english',
                                    ngram_range = (1,3),
                                    tokenizer = tokenizer.tokenize,
                                    max_features=10000)

    tfidf_matrix = tfidf_vectorizer.fit_transform(df_lsa["essay"])
    print(f"Train TFIDF Matrix Shape: {tfidf_matrix.shape}")

    # TFIDF to SVD
    svd_model = TruncatedSVD(n_components=100,
                            n_iter=200,
                            random_state=69)
        
    svd = svd_model.fit_transform(tfidf_matrix)

    end_vector = time.time()

    print(f"===Vector time: {end_vector - start_vector}===")

    print("Training Start")

    start_training = time.time()

    print("Getting Features")
    x_df_features = df_lsa[['word_count', 
                            'unique_word_count',
                            'sentence_count',
                            'avg_word_len',
                            'grammar_errors',
                            'lsa_score', 
                            'noun_count',
                            'adj_count',
                            'verb_count',
                            'adv_count']]

    x_features = np.concatenate((x_df_features.to_numpy(), svd), axis=1)
    y_features = df_lsa['actual_score'].to_numpy()

    print("Splitting Dataset")
    x_train, x_test, y_train, y_test = train_test_split(x_features, y_features, test_size = 0.2, train_size = 0.8, random_state = 420)

    print("Building Linear Regression Model")
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)

    print("Building SVR Model")
    svr_model = SVR()
    svr_model.fit(x_train, y_train)

    print("Building Decision Tree Model")
    tree_model = DecisionTreeRegressor()
    tree_model.fit(x_train, y_train)

    print("Building Bayesian Regressor")
    bayes_model = BayesianRidge()
    bayes_model.fit(x_train, y_train)

    print("Building AdaBoost Regressor")
    ada_model = AdaBoostRegressor(n_estimators=100)
    ada_model.fit(x_train, y_train)

    print("Building Random Forest Regressor")
    ran_model = RandomForestRegressor()
    ran_model.fit(x_train, y_train)

    print("Building Gradient Boosting Regressor")
    grad_model = GradientBoostingRegressor(n_estimators=200)
    grad_model.fit(x_train, y_train)

    print("Building Logistic Regression Model")
    log_model = LogisticRegression(solver="saga", max_iter=10000)
    log_model.fit(x_train, y_train)

    print("Getting Predictions")
    predictions = [ lr_model.predict(x_test),
                    svr_model.predict(x_test),
                    tree_model.predict(x_test),
                    bayes_model.predict(x_test),
                    ada_model.predict(x_test),
                    ran_model.predict(x_test),
                    grad_model.predict(x_test),
                    log_model.predict(x_test)]
    scores = []
    
    for idx, pred in enumerate(predictions):
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r_score = r2_score(y_test, pred)

        scores.append([idx, mae, mse, rmse, r_score])
    
    print("\nResults:")
    best_score = max(scores, key=lambda sublist: sublist[-1])
    print(f"Model {best_score[0]}")
    print(f"Mean Absolute Error: {best_score[1]}")
    print(f"Mean Squared Error: {best_score[2]}")
    print(f"Root Mean Squared Error: {best_score[3]}")
    print(f"R2 score: {best_score[4]}\n")

    print("Cross Validation 10-Folds")

    kf = KFold(n_splits=10)

    scores = [cross_val_score(lr_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(svr_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(tree_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(bayes_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(ada_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(ran_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(grad_model, x_features, y_features, cv=kf).mean(),
          cross_val_score(log_model, x_features, y_features, cv=kf).mean()]
    
    print(f"Model {scores.index(max(scores))}")
    print(f"Overall Score: {max(scores)}\n")

    end_training = time.time()

    print(f"===Training time: {end_training - start_training}===")

scorer_with_lsa_similarity()
end = time.time()
print(f"=-=-=-=overall exec time: {end - start}=-=-=-=")

