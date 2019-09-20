#!/usr/bin/env python

import nltk, inspect
import itertools
from nltk.corpus import brown
from nltk.tag import map_tag

from nltk.probability import ConditionalProbDist,ConditionalFreqDist, LidstoneProbDist

assert map_tag('brown', 'universal', 'NR-TL') == 'NOUN', '''
Brown-to-Universal POS tag map is out of date.'''

class HMM:
    def __init__(self, train_data, test_data):
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :param test_data: the test/evaluation dataset, a list of sentence with tags
        :type test_data: list(list(tuple(str,str)))
        """
        self.train_data = train_data
        self.test_data = test_data

        # Emission and transition probability distributions
        self.emission_PD: ConditionalProbDist = None
        self.transition_PD: ConditionalProbDist = None
        self.states = []

        self.viterbi = []
        self.backpointer = []

    # Compute emission model using ConditionalProbDist with the estimator:
    # Lidstone probability distribution with +0.01 added to the sample count for each bin and an extra bin
    def emission_model(self, train_data):
        """
        Compute an emission model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The emission probability distribution and a list of the states
        :rtype: Tuple[ConditionalProbDist, list(str)]
        """
    #   print(train_data)
        # TODO prepare data

        # Don't forget to lowercase the observation otherwise it mismatches the test data
        
        # I want to make train_data into one list of tagged_words with type:(tuple(str,str))
        data = []
        for x in train_data:
        #    data += [ (tag, word.lower() if word.isalpha() else (tag, word)) for (word, tag) in x]  # lower case and check word
            data += [ (tag, word.lower() )for (word, tag) in x]  # lower case

        # TODO compute the emission model
        emission_FD = ConditionalFreqDist(data)
        # need Lidstone bin parameter
        lidstone_estimator = lambda fd: LidstoneProbDist(fd, 0.01, fd.B() + 1)
        self.emission_PD = ConditionalProbDist(emission_FD, lidstone_estimator)
        
        self.states = emission_FD.keys()

        return self.emission_PD, self.states

    # Compute transition model using ConditionalProbDist with the estimator:
    # Lidstone probability distribution with +0.01 added to the sample count for each bin and an extra bin
    def transition_model(self, train_data):
        """
        Compute an transition model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The transition probability distribution
        :rtype: ConditionalProbDist
        """
        # TODO: prepare the data
        data = []
        tagged_sentences = train_data
        for s in tagged_sentences:

            data.append(("<s>",s[0][1]))          # s.insert(0, "<s>")
            for i in range(len(s)-1):
                data.append((s[i][1], s[i+1][1]))
            data.append((s[len(s)-1][1],"</s>"))  # s.insert((len(s)-1), "</s>")          
        
        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL <s> and the END SYMBOL </s>
      
   #     for s in tagged_sentences:
         #   (["<s>"] + s + ["</s>"])


        # tagged_sentencts: list(list(str or tuple))
       # tagGenerators=(((s[i][1],s[i+1][1]) for i in range(len(s)-1)) for s in tagged_sentences)

        # tagGenerators is an iterator of iterators of pairs of tags
      #  data = itertools.chain.from_iterable(tagGenerators)       

        # TODO compute the transition model

        transition_FD = ConditionalFreqDist(data)
        
        lidstone_estimator = lambda fd: LidstoneProbDist(fd, 0.01, fd.B() + 1)
        self.transition_PD = ConditionalProbDist(transition_FD, lidstone_estimator)
        
    #    print (data)

        return self.transition_PD
    
    def test_emission(self):
        print ("test emission")
        transition_PD = self.transition_model(self.train_data)
        t1 = -self.emission_PD['NOUN'].logprob('fulton')         # the printed result is positive = logprob return negatives
        print(t1)

    def test_transition(self):
        print ("test transition")
        transition_PD = self.transition_model(self.train_data)
        t1 = -transition_PD['<s>'].logprob('NOUN')
        t2 = -transition_PD['VERB'].logprob('</s>')
        t3 = -transition_PD['NOUN'].logprob('VERB')
        print(t1,t2,t3)
    
    # Train the HMM
    def train(self):
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part B: Implementing the Viterbi algorithm.

    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag
    def initialise(self, observation):
        """
        Initialise data structures for tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :type observation: str
        """
        
        # use costs (-log-base-2 probabilities)

        # viterbi is a list if libraries, i want it to store a T*N table 
        # viterbi[time step][ending state]         
        self.viterbi = [{}]
        
        # backpointer library
        self.backpointer = {}   # a list of tags
        
        # At initialise, cost with +logprob or *prob?
        for state in self.states:
           #  transition from <s> to observation
           # self.viterbi[0][state] = self.transition_PD["<s>"].prob(state) * self.transition_PD[state].prob(observation)
            
            self.viterbi[0][state] = self.transition_PD["<s>"].logprob(state) + self.emission_PD[state].logprob(observation)
            
            self.backpointer[state] = [state] 
        
        
    # Tag a new sentence using the trained model and already initialised data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer datastructures.
    # Describe your implementation with comments.
    # Input: list of words
    def tag(self, observations):
        """
        Tag a new sentence using the trained model and already initialised data structures.

        :param observations: List of words (a sentence) to be tagged
        :type observations: list(str)
        :return: List of tags corresponding to each word of the input
        """
        tags = []
        index = 0
        current_decision = []             #   0.8959
 
        for t in range(1, len(observations)):
            self.viterbi.append({})
            newbackpointer = {}   # index-state key-the list of my states
            
            for state in self.states:

                # viterbi min
                # logprob return negative numbers, adding more negative value => want a result closer to + => max
                
                (prob,prob_state) = max(
                    [(self.viterbi[t-1][previous_state] + self.transition_PD[previous_state].logprob(state) 
                      + self.emission_PD[state].logprob(observations[t]),previous_state) for previous_state in self.states])                
                    
                self.viterbi[t][state] = prob     # update the probability

                # backpointer
                newbackpointer[state] = self.backpointer[prob_state] + [state]
                
            self.backpointer = newbackpointer 

        # Return the tag sequence corresponding to the best path as a list.
        
        # viterbi[time step][ending state] 
        # choose the best match
        
        # termination cost
        for state in self.states:
            self.viterbi[len(observations) -1][state] += self.transition_PD["<\s>"].logprob(state)
     #   print(self.viterbi[len(observations) -1][state])   # like -200
  
        (probability,state) = max([(self.viterbi[len(observations) -1][state], state) for state in self.states])
                                       
        # Reconstruct the tag sequence using the backpointer list.
        # Return the tag sequence corresponding to the best path as a list.
        tags = self.backpointer[state]
        
        return tags        

def answer_question4b():
    """ Report a tagged sequence that is incorrect
    :rtype: str
    :return: your answer [max 280 chars]"""
 
    # entries saves the first 10 incorrectly tagged sentences with their correct version
    entries = []    
    incorrect = []
    for sentence in test_data_universal:
        s = [word.lower() for (word, tag) in sentence]
        model.initialise(s[0])
        tags = model.tag(s)
        
        # append till 10 entries
        for ((word,gold),tag) in zip(sentence,tags):
            if tag != gold:
                incorrect.append(tags)
                entries.append(sentence)
                break
                
        if(len(entries) >= 10):  
            break
        
    # to print the zip file
  # print('length of entries is %f\n'%len(entries))  
  # for e,o in zip(entries,incorrect):
  #     print(e)
  #     print(o)
    
    tagged_sequence = incorrect[0]
    correct_sequence = entries[0]  
    
    # Why do you think the tagger tagged this example incorrectly?
    answer =  inspect.cleandoc("The Fulton County, if tagged correctly, should be DET NOUN NOUN, while tagger returns DET ADJ NOUN. Fulton is tagged wrong because it is tagged highly based on its previous word, however, in this case Fulton County should be regarded as a multiple-word phrase with both being NOUN")[0:280]

    return tagged_sequence, correct_sequence, answer

def answer_question5():
    """Suppose you have a hand-crafted grammar that has 100% coverage on
        constructions but less than 100% lexical coverage.
        How could you use a POS tagger to ensure that the grammar
        produces a parse for any well-formed sentence,
        even when it doesn't recognise the words within that sentence?

    :rtype: str
    :return: your answer [max 500 chars]"""

    return inspect.cleandoc("Using Lidstone and negative log probability, the model returns us the tag which is most probable given the tag of the previous word. If my tagger does recognize the word, it probably does better because we analyze the current state by transition and emission model. It may not do better if there is ambiguity of which tag the word could have. Given a word unrecognized, my approach might not be better because we cannot infer the state with emission model being considered P(o|state).")[0:500]


# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def answers():
    global tagged_sentences_universal, test_data_universal, \
           train_data_universal, model, test_size, train_size, ttags, \
           correct, incorrect, accuracy, \
           good_tags, bad_tags, answer4b, answer5
    
    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news', tagset='universal')

    # Divide corpus into train and test data.
    test_size = 1000
    train_size = len(tagged_sentences_universal) - 1000

    test_data_universal = tagged_sentences_universal[:test_size]
    train_data_universal = tagged_sentences_universal[-train_size:]

    # Create instance of HMM class and initialise the training and test sets.
    model = HMM(train_data_universal, test_data_universal)

    # Train the HMM.
    model.train()

    # Inspect the model to see if emission_PD and transition_PD look plausible
    print('states: %s\n'%model.states)
    # Add other checks

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    
    # self designed tests
    
 #   model.test_emission()
    
 #   model.test_transition()
    
    s='the cat in the hat came back'.split()
  #  s='Tell me olo cat is gooood'.split()
    model.initialise(s[0])
    ttags = model.tag(s) 
    print("Tag a trial sentence")
    print(list(zip(s,ttags)))

    # check the model's accuracy (% correct) using the test set
    correct = 0
    incorrect = 0

    for sentence in test_data_universal:
        s = [word.lower() for (word, tag) in sentence]
        model.initialise(s[0])
        tags = model.tag(s)

        for ((word,gold),tag) in zip(sentence,tags):
            if tag == gold:
                correct += 1
            else:
                incorrect += 1

    accuracy = (correct / (correct + incorrect))
    print('Tagging accuracy for test set of %s sentences: %.4f'%(test_size,accuracy))

   

    # Print answers for 4b and 5
    bad_tags, good_tags, answer4b = answer_question4b()
    print('\nAn incorrect tagged sequence is:')
    print(bad_tags)
    print('The correct tagging of this sentence would be:')
    print(good_tags)
    print('\nA possible reason why this error may have occurred is:')
    print(answer4b[:280])
    answer5=answer_question5()
    print('\nFor Q5:')
    print(answer5[:500])

if __name__ == '__main__':
    answers()
