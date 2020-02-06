'''
LDA Project

In order to use this file, the database has first to be created, utilizing the script extract_ingredients.py (done)

food101_recipes_with_reduced_ingrs.json is a processed subset of Recipe1M. We extract recipes out of Recipe1M that
share the meal class with the well known food101 dataset. The set-union is a composition of 80 classes. For each of
this classes, we provide at max 50 recipes.

- evaluate the topic distributions for recipes
    1) plot histogram,
    2) calculate distribution similarity (Kullback-Leibler-Divergence) between recipes of the same food101 classes
            (We will gain insight if LDA sorts similar than humans do if meals of the same class are assigned to
            the same topics.)
    3) calculate entropy and compare it to the entropy of a uniform distribution
    4) If we use less topics than meal classes, what meal classes are represented by the same topic or by a similar
       topic distribution

- evaluate the ingredient distribution (word distribution)
    1) plot histogram,
    2) calculate distribution similarity (Kullback-Leibler-Divergence) and compare the word-distributions against
       each other. (They should be dissimilar)
    3) calculate entropy and compare it to the entropy of a uniform distribution.

- improve the quality of the topic distribution by changing the number of latent topics

- improve the ingredient distribution by excluding words that are shared over several topics. (we would like to have
  unique word distributions for each topic)
'''

import numpy as np
import simplejson as json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA


def KLD(dist_1, dist_2, eps=1e-8):
    '''
    dist_1 a numpy array of probabilities
    dist_2 a numpy array of probabilities
    KLD = 0 id dist_1 = distr_2
    '''
    dist_1 = dist_1 + eps
    dist_2 = dist_2 + eps
    return np.sum(dist_1 * np.log(dist_1/dist_2))

def ShanonEntropy(p):
    '''
    p = a numpy array of probabilities
    Shanon Entropy is large for uniform distributions
    '''
    return -np.sum(p*np.log(p))

# load the recipe database
with open('food101_recipes_with_reduced_ingrs.json') as f_layer:
    food101_recipes_with_reduced_ingrs =  json.load(f_layer)

# extract igredeitns information
just_ingrs = [' '.join(rec['ingredients']) for rec in food101_recipes_with_reduced_ingrs]

#represent the ingredients as a Bag of Words
count_vectorizer = CountVectorizer()
BoW = count_vectorizer.fit_transform(just_ingrs).toarray()

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'magenta', 'cyan', 'red', 'orange', 'yellow']
k = 0

print(len(just_ingrs))

number_topics = [1,5,10,25]

# Create and fit the LDA model
number_words = 15  # number of words we are going to plot
for i in number_topics:
    lda = LDA(n_components=i, n_jobs=-1)  # initialize LDA
    lda.fit(BoW)  # fit the data to the model
    
    # list the unique vocabulary created by the CountVectorizer
    words = count_vectorizer.get_feature_names()
    
    # generate word distributions utilizing the trained LDA-model
    word_dists = lda.components_ / np.sum(lda.components_, axis=1, keepdims=True)
    
    
    most_common_words_per_topic = []
    for topic_idx, topic in enumerate(word_dists):
        print('\nTopic #{}:'.format(topic_idx))
        most_common_word_index = topic.argsort()[::-1][:number_words]
        most_common_words_per_topic.append([words[index] for index in most_common_word_index])
        output = ', '.join(['{0}'.format(words[index]) for index, contribution in
                             zip(most_common_word_index, topic[most_common_word_index])])
        print(output)
    
    # get the topic distribution for the first 20 recipes
    for i in range(20):
        topic_id = np.argmax(lda.transform([BoW[i,:]]))
        print('\nTitle: {0}'
              '\nClass: {1}'
              '\nTopic ID: {2} ({3:.2f}%)'
              '\nIngredients: {4}'
              '\nMost common words: {5}'.format(food101_recipes_with_reduced_ingrs[i]['title'],
                                                food101_recipes_with_reduced_ingrs[i]['class'],
                                                topic_id,
                                                lda.transform([BoW[i, :]])[0,topic_id]*100,
                                                food101_recipes_with_reduced_ingrs[i]['ingredients'],
                                                ' '.join(most_common_words_per_topic[topic_id])))
    
    # get and print the first recipe's topic distribution
    print('\nTopicdistribution\n{0}'.format(lda.transform([BoW[0,:]])[0]))
    
    #Plot histogram with most likely topic
    topic_list = []
    for j in range(len(just_ingrs)):
        topic_id = np.argmax(lda.transform([BoW[j,:]]))
        topic_list.append(lda.transform([BoW[j, :]])[0,topic_id]*100)
    num_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    y = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    n, bins = np.histogram(topic_list, bins=num_bins)
    plt.plot(y, n, color=colors[k], marker='.')
    k = k + 1

plt.title('Amount of recipes with maximum probability for a topic, given different number of topics')
plt.xlabel('Probability')
plt.ylabel('Nº of recipes with a certain probability')
plt.show()
        