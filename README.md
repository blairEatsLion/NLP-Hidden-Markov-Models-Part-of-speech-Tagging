# NLP-Hidden-Markov-Models-Part-of-speech-Tagging
Using NLTK, train a HMM model and return POS tags by Viterbi algorithm

Part 1: TRAINING A HIDDEN MARKOV MODEL
Create and train two models - an Emission Model and a Transition Model
- Clean and structure data
- Use ConditionalProbDist with a LidstoneProbDist estimator +0.01 added to the sample count for each bin.

Part 2: IMPLEMENTING THE VITERBI ALGORITHM (for POS tagging)
Implement the Viterbi algorithm
- 2 data structures for the algorithm: the viterbi data structure and the backpointer data structure
- use costs (negative log probabilities)
- recursion and termination step of algorithm
- test the algorithm
