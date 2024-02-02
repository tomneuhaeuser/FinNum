import torch
import random
import numpy as np

import json
import re
import preprocessor
import pycorenlp
import nltk

import time

# Reproducability
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Pad the tweets and labels in a batch such that they will all be of equal length
def collate_fn(items):
    tweets, labels, lengths = zip(*items)
    
    maximum_length = max(lengths)
    
    tweets_padded = torch.zeros((len(items), maximum_length, 101))
    labels_padded = torch.zeros((len(items), maximum_length))
    
    for i, (tweet, label, length) in enumerate(zip(tweets,labels,lengths)):
        offset = int((maximum_length - length)/2)
        tweets_padded[i][offset:(offset+length)] = tweet
        labels_padded[i][offset:(offset+length)] = label

    return tweets_padded.to(torch.float32), labels_padded.to(torch.long)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset = 'train'):
        with open(dataset + '.json', 'r') as f:
                self.data = json.load(f)
                
        # Set tweet prepocessor options
        preprocessor.set_options(preprocessor.OPT.URL,
                                 preprocessor.OPT.HASHTAG,
                                 preprocessor.OPT.MENTION,
                                 preprocessor.OPT.EMOJI, 
                                 preprocessor.OPT.SMILEY)
                                 
        # Initialize StanfordCoreNLP (java -Xmx5g edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000)
        self.nlp = pycorenlp.StanfordCoreNLP('http://localhost:9000')
        
        # Initialize the lookup table to vectorize tokens
        self.lemma_lookup_table = {}
        
        with open('glove.6B.50d.txt', 'r', encoding = 'utf-8') as f:
            for l in f:
                s = l.split()
                self.lemma_lookup_table[s[0]] = np.asarray(s[1:], 'float32')
        
        # Initialize the lookup tabe to one-hot encode pos tags
        self.pos_lookup_table = {'CC':   np.asarray([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'CD':   np.asarray([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'DT':   np.asarray([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'EX':   np.asarray([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'FW':   np.asarray([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'IN':   np.asarray([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'JJ':   np.asarray([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'JJR':  np.asarray([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'JJS':  np.asarray([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'LS':   np.asarray([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'MD':   np.asarray([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'NN':   np.asarray([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'NNS':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'NNP':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'NNPS': np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'PDT':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'POS':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'PRP':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'PRP$': np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'RB':   np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'RBR':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'RBS':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'RP':   np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'SYM':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'TO':   np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'UH':   np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'VB':   np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'VBD':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], 'float32'),
                                 'VBG':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], 'float32'),
                                 'VBN':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], 'float32'),
                                 'VBP':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], 'float32'),
                                 'VBZ':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], 'float32'),
                                 'WDT':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], 'float32'),
                                 'WP':   np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], 'float32'),
                                 'WP$':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], 'float32'),
                                 'WRB':  np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], 'float32')}
        
        # Initialize the lookup table to one-hot encode named entities
        self.ner_lookup_table = {'O':            np.asarray([1,0,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'PERSON':       np.asarray([0,1,0,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'LOCATION':     np.asarray([0,0,1,0,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'ORGANIZATION': np.asarray([0,0,0,1,0,0,0,0,0,0,0,0,0], 'float32'),
                                 'MISC':         np.asarray([0,0,0,0,1,0,0,0,0,0,0,0,0], 'float32'),
                                 'MONEY':        np.asarray([0,0,0,0,0,1,0,0,0,0,0,0,0], 'float32'),
                                 'NUMBER':       np.asarray([0,0,0,0,0,0,1,0,0,0,0,0,0], 'float32'),
                                 'ORDINAL':      np.asarray([0,0,0,0,0,0,0,1,0,0,0,0,0], 'float32'),
                                 'PERCENT':      np.asarray([0,0,0,0,0,0,0,0,1,0,0,0,0], 'float32'),
                                 'DATE':         np.asarray([0,0,0,0,0,0,0,0,0,1,0,0,0], 'float32'),
                                 'TIME':         np.asarray([0,0,0,0,0,0,0,0,0,0,1,0,0], 'float32'),
                                 'DURATION':     np.asarray([0,0,0,0,0,0,0,0,0,0,0,1,0], 'float32'),
                                 'SET':          np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,1], 'float32')}
        
        # Initialize the lookup table to one-hot encode labels
        self.label_lookup_table = {'O':              0,
                                   'Monetary':       1,
                                   'Percentage':     2,
                                   'Option':         3,
                                   'Indicator':      4,
                                   'Temporal':       5,
                                   'Quantity':       6,
                                   'Product Number': 7}
    
    # Preprocess the tweets
    def __get_tweet__(self, tweet):
        tweet_preprocessed = tweet
        
        # Remove tab and newline characters
        tweet_preprocessed = re.sub(r'\t|\n', ' ', tweet_preprocessed)
            
        # Remove HTML entities
        tweet_preprocessed = re.sub(r'\&[^;]*;', '', tweet_preprocessed)
        
        # Remove URLs, hashtags, mentions, emojis and smileys
        tweet_preprocessed = preprocessor.clean(tweet_preprocessed)
        
        # Replace cashtags by 'stock'
        tweet_preprocessed = re.sub(r'\$[^0-9\.\s]+', 'stock', tweet_preprocessed)
        
        # Replace '%' by 'RFPC' (Replacement For Percentage Character) since StanfordCoreNLP can not handle '%'
        tweet_preprocessed = re.sub(r'%', 'RFPC', tweet_preprocessed)
        
        # Remove all special characters except '.', ',', ':', '/' and '$'
        tweet_preprocessed = re.sub(r'[^a-zA-Z0-9\.,:/\$ ]', ' ', tweet_preprocessed)
        
        # Add a '0' before a '.' followed but not preceeded by a digit (e.g. '.1' -> '0.1')
        tweet_preprocessed = re.sub(r'(?<!\d)\.\d+', r'0\g<0>', tweet_preprocessed)
        
        # Remove all occurences of '.', ',', ':', '/' except if they occur between digits
        tweet_preprocessed = re.sub(r'(?<!\d)(\.|,|:|/)(?!\d)', ' ', tweet_preprocessed)
        
        # Uncouple numbers from other characters (e.g. '15th' -> '15 th' or '1M' -> '1 M')
        tweet_preprocessed = re.sub(r'\d+(\.|,)\d+|\d+', r' \g<0> ', tweet_preprocessed)
        
        # Replace multiple space characters by one space character
        tweet_preprocessed = re.sub(r'\s+', ' ', tweet_preprocessed)
        
        # Transform all characters to lower case
        tweet_preprocessed = tweet_preprocessed.lower()
        
        return tweet_preprocessed
       
    # Preporcess the target numerals 
    def __get_target_num__(self, target_num):
        target_num_preprocessed = []
        
        for tn in target_num:
            if tn[0] == '.' and tn[-1] == '.':
                target_num_preprocessed.append(re.sub(r'(?<!\d)\.\d+', r'0\g<0>', tn[:-1]))
            elif tn[0] == '.':
                target_num_preprocessed.append(re.sub(r'(?<!\d)\.\d+', r'0\g<0>', tn))
            elif tn[-1] == '.':
                target_num_preprocessed.append(tn[:-1])
            else:
                target_num_preprocessed.append(tn)
        
        return target_num_preprocessed
        
    def __get_features__(self, tweet):
        tweet_annotated = self.nlp.annotate(tweet, properties={'annotators': 'ner, pos', 'outputFormat': 'json', 'timeout': 10000})
        
        lemma = []
        pos = []
        ner = []
        
        for sentence in tweet_annotated["sentences"]:
            for token in sentence["tokens"]:
                if token['word'] != '.' and token['word'] != ',' and token not in nltk.corpus.stopwords.words('english'):
                    # Re-Replace 'RFPC' (Replacement For Percentage Character) by '%'
                    if token['word'] == 'rfpc':
                        lemma.append('%')
                        pos.append('SYM')
                        ner.append('PERCENT')
                    else:
                        lemma.append(token['lemma'])
                        pos.append(token['pos'] if token['pos'] in self.pos_lookup_table.keys() else 'SYM')
                        ner.append(token['ner'] if token['ner'] in self.ner_lookup_table.keys() else 'O')
        
        for i, n in enumerate(ner):
            if i >= 1 and ner[i] == 'PERCENT' and ner[i-1] == 'NUMBER':
                ner[i-1] = 'PERCENT'

        return lemma, pos, ner

    def __find_positions__(self, lemma, target_num):
        positions = []
        
        for tn in target_num:
            positions.extend([i for i in range(len(lemma)) if lemma[i] == tn])
        
        positions = list(set(positions))
        positions.sort()
        
        assert len(target_num) == len(positions)
        return positions
    
    def __get_position_and_label__(self, lemma, target_num, category):
        positions = self.__find_positions__(lemma, target_num)
    
        position = ['O']*len(lemma)
        label = ['O']*len(lemma)

        for p, c in zip(positions,category):
            position[p] = 'X'
            label[p] = c

        return position, label
    
    # Preprocess all numerals in a tweet (to reduce the OOV Rate)
    def __preprocess_numerals__(self, lemma):
        for i, l in enumerate(lemma):
            if re.match(r'\d+\.\d+$', l):
                s = l.split('.')
                if len(s[0]) == 1 and s[0] == ['0']:
                    lemma[i] = '0.00'
                else:
                    lemma[i] = '1' + '0'*(min(len(s[0])-1,3)) + '.' + '00'
            elif re.match(r'\d+,\d+$', l):
                lemma[i] = '1' + '0'*(min(len(l)-2,3))
            elif re.match(r'\d+$', l):
                lemma[i] = '1' + '0'*(min(len(l)-1,3))
        
        return lemma
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        tweet = self.__get_tweet__(self.data[index]['tweet'])  
        target_num = self.__get_target_num__(self.data[index]['target_num'])
        category = self.data[index]['category']
        
        lemma, pos, ner = self.__get_features__(tweet)
        
        position, label = self.__get_position_and_label__(lemma, target_num, category)
        
        lemma = self.__preprocess_numerals__(lemma)
        
        # Vectorize tweet
        tweet_vectorized = np.zeros((len(lemma),101))
        
        for i,(l,p,n) in enumerate(zip(lemma, pos, ner)):
            try:
                l_vectorized = self.lemma_lookup_table[l]
            except:
                l_vectorized = self.lemma_lookup_table['unk']
            else:
                p_vectorized = self.pos_lookup_table[p]
                n_vectorized = self.ner_lookup_table[n]
                
                tweet_vectorized[i] = np.concatenate((l_vectorized, p_vectorized, n_vectorized, torch.ones(1) if position[i] == 'X' else torch.zeros(1), torch.ones(1)))
        
        # Vectorize label
        label_vectorized = np.zeros(len(label))
        
        for i,l in enumerate(label):
            label_vectorized[i] = self.label_lookup_table[l]
                
        return torch.from_numpy(tweet_vectorized).to(torch.float32), torch.from_numpy(label_vectorized).to(torch.long), len(label)