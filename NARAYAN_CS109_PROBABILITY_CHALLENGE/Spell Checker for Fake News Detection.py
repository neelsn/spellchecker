#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv # read from file
import math # exponentiation
import string # string processing
import nltk # get corpus
import time # efficiency
import heapq # 'top k'
import pandas as pd # csv processing
import numpy as np # numerical processing
from operator import itemgetter # sorting
from english_words import english_words_set # extra words


# In[2]:


nltk.download('words')
from nltk.corpus import words


# In[3]:


df = pd.read_csv("True.csv")

words = []
words.append('<s>')
for i in range(len(df) - 1):
    for word in df.iloc[i][1].split():
        words.append(word.lower())
        if '.' in word and word.index('.') == (len(word) - 1):
            words.append('</s>')

print(len(words))


# In[4]:


# print sample article
print(df.iloc[1][1])


# In[5]:


def create_bigrams(words):
    bigrams = []
    bigrams_freq = {}
    word_freq = {}
    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        if i < len(words) - 1:
            bigrams.append(bigram)
            if bigram in bigrams_freq:
                bigrams_freq[bigram] += 1
            else:
                bigrams_freq[bigram] = 1
        if words[i] in word_freq:
            word_freq[words[i]] += 1
        else:
            word_freq[words[i]] = 1
    return bigrams, word_freq, bigrams_freq


# In[6]:


bigrams, word_freq, bigrams_freq = create_bigrams(words)


# In[7]:


def calculate_bigrams_probability(bigrams, word_freq, bigrams_freq):
    probabilities = {}
    for bigram in bigrams:
        word1 = bigram[0]
        probabilities[bigram] = math.log((bigrams_freq[bigram] + 1) / (word_freq[word1] + 8579806))
    return probabilities


# In[8]:


bigram_probabilities = calculate_bigrams_probability(bigrams, word_freq, bigrams_freq)


# In[9]:


def calculate_unknown_bigrams_probability(bigram):
    if bigram[0] not in word_freq:
        word_freq[bigram[0]] = 1
    bigram_probabilities[bigram] = math.log(1 / (word_freq[bigram[0]] + 8579806))


# In[10]:


bad_bigram = ('trump', 'donald')
if bad_bigram not in bigrams:
    calculate_unknown_bigrams_probability(bad_bigram)
print(bigram_probabilities[bad_bigram])
good_bigram = ('donald', 'trump')
print(bigram_probabilities[good_bigram])


# In[11]:


# prints most common bigram
print(max(bigram_probabilities, key = bigram_probabilities.get))

# prints top 25 bigrams from the article with the largest probabilities
largest_25 = heapq.nlargest(25, bigram_probabilities, key = bigram_probabilities.get)
for l in largest_25:
    print(l, ": ", bigram_probabilities[l])


# In[12]:


# test sentence
original_sentence = "The former president, Donald Frump, lived in the White House."


# In[13]:


def format_sentence(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation)).lower()

sentence = format_sentence(original_sentence)
print(sentence)


# In[14]:


sentence_words = sentence.split()
s_bigrams, s_word_freq, s_bigrams_freq = create_bigrams(sentence_words)


# In[15]:


print(s_bigrams)


# In[16]:


print(s_word_freq)


# In[17]:


print(s_bigrams_freq)


# In[18]:


s_bigrams_probabilities = {}
for bigram in s_bigrams:
    if bigram not in bigram_probabilities:
        calculate_unknown_bigrams_probability(bigram)
    s_bigrams_probabilities[bigram] = bigram_probabilities[bigram]

print(s_bigrams_probabilities)


# In[19]:


smallest_probability = min(s_bigrams_probabilities, key=s_bigrams_probabilities.get)
print(smallest_probability)


# In[20]:


two_smallest = sorted(s_bigrams_probabilities.items(), key=itemgetter(1))[:2]

print(two_smallest)


# In[21]:


incorrect_word = two_smallest[0][0][1]
# incorrect_word = two_smallest[1][0][0]

print(incorrect_word)


# In[22]:


english_words = list(english_words_set)


# In[23]:


print('hello' in english_words)
print('trump' in english_words)
print('clinton' in english_words)
print('obama' in english_words)
print('jump' in english_words)
print('esoteric' in english_words)


# In[24]:


def minimum_edit_distance(word1, word2):
    # levenshtein distance minimum edit distance table
    amt_rows = len(word1) + 1
    amt_cols = len(word2) + 1
    edit_distance = np.zeros((amt_rows, amt_cols))

    for r in range(1, amt_rows):
        edit_distance[r][0] = r
    for c in range(1, amt_cols):
        edit_distance[0][c] = c

    for row in range(1, amt_rows):
        for col in range(1, amt_cols):
            if word1[row - 1] == word2[col - 1]:
                edit_distance[row][col] = edit_distance[row - 1][col - 1]
            else:
                edit_distance[row][col] = min(edit_distance[row][col - 1] + 1,
                                              edit_distance[row - 1][col] + 1,
                                              edit_distance[row - 1][col - 1] + 2)

    return edit_distance[amt_rows - 1][amt_cols - 1]


# In[25]:


print(minimum_edit_distance('apple', 'trample'))


# In[26]:


print(len(english_words))


# In[27]:


distances = {}
for i in range(len(english_words) - 1):
    distances[english_words[i]] = minimum_edit_distance(english_words[i], incorrect_word)

twenty = sorted(distances.items(), key=itemgetter(1))[:20]

print(twenty[0:len(twenty)-1])


# In[28]:


first_word = two_smallest[0][0][0]
new_probabilities = {}

for i in range(len(twenty)):
    bigram = (first_word, twenty[i][0])
    if bigram not in bigram_probabilities:
        calculate_unknown_bigrams_probability(bigram)
    new_probabilities[bigram] = bigram_probabilities[bigram]

largest = sorted(new_probabilities.items(), key=itemgetter(1))[-1:]

print(largest)
correct_word = largest[0][0][1]
print(correct_word)


# In[29]:


print("Original Sentence:", original_sentence)
print("Did you mean to replace", incorrect_word, "with", correct_word, "?")
response = input("")

final_sentence = original_sentence

if response == "yes":
    for word in original_sentence.translate(str.maketrans('', '', string.punctuation)).split():
        if word.lower() == incorrect_word:
            final_sentence = final_sentence.replace(word, correct_word)
            break
    print("Updated Sentence: ", final_sentence)
else:
    print("We could not find a suitable change for", incorrect_word)


# In[30]:


print('reporter' in english_words)


# In[31]:


# trie to make searching faster


# In[32]:


class Trie:
    def __init__(self):
        self.word = None
        self.children = {}
    def add(self, word):
        node = self
        for l in word:
            if l not in node.children: # only add letter if it isn't already there
                node.children[l] = Trie()
            node = node.children[l]
        node.word = word

def find(node, word, letter, previous, size, words):
    row = [previous[0] + 1]

    for column in range(1, len(word) + 1):
        delete = previous[column] + 1 # deletions cost 1
        insert = row[column - 1] + 1 # insertions cost 1
        replace = 0
        
        if word[column - 1] == letter:
            replace = previous[column - 1]
        else:                
            replace = previous[column - 1] + 2 # substitutions cost 2
        row.append(min(delete, insert, replace)) 
    if min(row) <= size:
        for letter in node.children:
            find(node.children[letter], word, letter, row, size, words)
    if node.word != None and row[len(row) - 1] <= size:
        words.append((node.word, row[len(row) - 1]))

def search(word, size):
    row = range(len(word) + 1)
    words = []
    for l in trie.children:
        find(trie.children[l], word, l, row, size, words) # recursively search
    return words

start = time.time()
trie = Trie()
for w in english_words:
    trie.add(w)
words = search("frump", 3)
end = time.time()
print(words)
print("Search took %g seconds" % (end - start))

