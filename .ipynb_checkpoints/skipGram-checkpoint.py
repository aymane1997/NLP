from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


__authors__ = ['author1','author2','author3']
__emails__  = ['fatherchristmoas@northpole.dk','toothfairy@blackforest.no','easterbunny@greenfield.de']

def text2sentences(path):
# feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( l.lower().split() )
    return sentences

def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


# +
class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.trainset = {sentence for sentence in sentences} # set of sentences
        self.w2id = {} # word to ID mapping
        self.vocab = set() # list of valid words
        self.w2weight = {}
        
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount
        self.nEmbed = nEmbed
        
        for sentence in self.trainset:
            for word in sentence:
                if word not in self.w2id:  
                    self.w2id[word]= len(self.w2id)+1
                    self.vocab.add(word)    
                word_id = self.w2id[word] 
                if word_id in self.w2weight : 
                    self.w2weight[word_id]+=1
                else :
                    self.w2weight[word_id]=1
        
        for word in self.vocab :
            if self.w2weight[self.w2id[word]] <= self.minCount : 
                self.vocab.discard(word)
        
        self.target = np.zeros((self.nEmbed, len(self.vocab)))
        self.context = np.zeros((self.nEmbed, len(self.vocab)))
        self.loss = []
        self.trainWords = 0
        self.accLoss = 0.
        self.lr = 0.005 
      
        
     
    def sample(self, omit):

        # generate randomly according to weights using A-Chao WeightedReservoir algorithm
        
        omit_ids =  {self.w2id[word_to_omit] for word_to_omit in omit}   
        WSum = 0
        R = np.zeros(self.negativeRate)
        
        i = 0
        for word_id in enumerate(self.w2weight):
            if ((word_id in omit_ids): continue
            WSum += self.w2weight[word_id]**(3/4)
            
            # fill the reservoir array
            if i < self.negativeRate : 
                R[i] = word_id
                i+=1
            else : 
                p =  self.w2weight[word_id]**(3/4) / WSum  # probability for this item
                j = np.random.rand();                      # uniformly random between 0 and 1
                if j <= p :                                  # select item according to probability
                    R[np.random.randint(0, self.negativeRate)]  = word_id 
        return R

    def train(self):
        for counter, sentence in enumerate(self.trainset):
            sentence = filter(lambda word: word in self.vocab, sentence)

            for wpos, word in enumerate(sentence):
                wIdx = self.w2id[word]
                winsize = np.random.randint(self.winSize) + 1
                start = max(0, wpos - winsize)
                end = min(wpos + winsize + 1, len(sentence))

                for context_word in sentence[start:end]:
                    ctxtId = self.w2id[context_word]
                    if ctxtId == wIdx: continue
                    negativeIds = self.sample({wIdx, ctxtId})
                    self.trainWord(wIdx, ctxtId, negativeIds)
                    self.trainWords += 1

            if counter % 1000 == 0:
                print(' > training %d of %d' % (counter, len(self.trainset)))
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0
                self.accLoss = 0.

    def trainWord(self, wordId, contextId, negativeIds):
        t = self.target[:, wordId: wordId+1]
        c = self.context[:, contextId + list(negativeIds)]
        
        probabilties = (c.T).dot(t)
        probabilties = np.exp(probabilties)
        probabilties = normalize(probabilties)
        
        self.accLoss += -np.log(probabilties[0,0])
                
        self.target[:, wordId: wordId + 1] = t - \
                self.lr * (- c[:,0:1] + (c * probabilties.T).sum(axis = 1)[:, None])
        self.context[:, contextId : contextId] = c[:0:1] - \
                self.lr * ( (-1 + probabilties[0,0]) * t )
        for i in range(self.negativeRate): 
                self.context[:, negativeIds[i]: negativeIds[i]+1] = self.context[:, negativeIds[i]: negativeIds[i]+1] - \
                self.lr * ((-1 + probabilties[i+1,0]) * t)

    def save(self,path):
        raise NotImplementedError('implement it!')

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
       # to implement

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')
        

# -

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train(...)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a,b))

