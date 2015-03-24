from math import log, exp
import numpy
import re
import addmath
import Viterbi

discrete_distribution = addmath.discrete_distribution
eln = addmath.eln
exp = addmath.eexp
log_product = addmath.log_product
iter_plog = addmath.iter_plog

class homopolymer:
    def __init__(self, base='-', length=0):
        self.base = base
        self.length = length

class ViterbiVariable:
    def __init__(self, probability=0, path=[]):
        # records in path seems like(hp_reference, hp_read, state)
        self.probability = probability
        path = path

class HmmRecord:

    def __init__(self, baseCall, lengthCall, lengthCallInsertion, transitionProbabilities):
        """
        Constructor
        :param baseCallVectors: dictionary, three key: Deletion (value - probability of remove reference base),
        Insertion(value - probability of insert read base), Match (value - 1?)
        :param lengthCallInsertion: dictionaty, key - bases, values - length of gomopolymers of such bases when
        insertions occur
        :param transitionProbabilities: dictionary with trans.prob. for every state
        :return:
        """
        self.bases = ['A', 'C', 'G', 'T']
        self.states = ['Match', 'Deletion', 'Insertion', 'End', 'Begin']
        self.__emissionInsertionBaseProbabilities = baseCall['-']
        self.__emissionInsertionLengthProbabilities = {'A': lengthCallInsertion['A'], 'C': lengthCallInsertion['C'],
                                                       'G': lengthCallInsertion['G'], 'T': lengthCallInsertion['T']}
        # lengthCall['base']: matrix, row - input length, column - output
        self.__emissionMatchProbabilities = {'A': lengthCall['A'], 'C': lengthCall['C'], 'G': lengthCall['G'],
                                             'T': lengthCall['T']}
        # if transProb = {'base': [probabilities M, I, D]
        self.__transition_probabilities = transitionProbabilities

    def __emissionNotViterbi(self, h, state):
        """
        :param h: input homopolymer
        :param state: current state
        :return: emission homopolymer, when we are in state 'state' and there is input homopolymer h
        """
        if state == 'Deletion':
            return homopolymer()
        if state == 'Begin':
            return homopolymer()
        if state == 'Match':
            #[h.length + 1] - count from 0
            bins = numpy.cumsum(self.__emissionMatchProbabilities[h.base][h.length - 1])
            length = 0
            while length == 0:
                length = numpy.digitize(numpy.random.random_sample(1), bins)[0]
            return homopolymer(h.base, length)
        if state == 'Insertion':
            bins = numpy.cumsum(self.__emissionInsertionBaseProbabilities)
            base = self.bases[numpy.digitize(numpy.random.random_sample(1), bins)]
            length = 0
            while length == 0:
                bins = numpy.cumsum(self.__emissionInsertionLengthProbabilities[base])
                # '+ 1' - we count from 0
                length = numpy.digitize(numpy.random.random_sample(1), bins)[0] + 1
            return homopolymer(base, length)

    def __emissionViterbi(self, h, g, state):
        """
        :param h: input homopolymer
        :param g: output homopolymer
        :return: probability that state 'state' emits g with given h
        """
        if state == 'Begin':
            return 1
        if state == 'Deletion':
            return 1
        if state == 'Match':
            # [g.length - 1] - count from 0
            return self.__emissionMatchProbabilities[h.base][h.length][g.length - 1]
        if state == 'Insertion':
            # [g.length - 1] - count from 0
            return (self.__emissionInsertionBaseProbabilities[self.bases.index(g.base)] *
                   self.__emissionInsertionLengthProbabilities[g.base][g.length - 1])

    def emission(self, *args):
        """
        return emission probabilty
        :param args: in case 2 args - homopolymer and state, returns output homopolymer, accordingly to emission probs.
        In case 3 srgs - two HP and state, return probability of observing such output and input
        :return: homopolymer (case 2 args)or probability(case 3 args)
        """
        if len(args) == 2:
            # args = g, state. Need to know homopolymer output
            return self.__emissionNotViterbi(*args)
        if len(args) == 3:
            # args = g, h, state. Want to know probability
            return self.__emissionViterbi(*args)

    def transition(self, *args):
        """
        stateCurr, stateNext
        return transition probability of these two states. From current to next, or models next state
        :param stateCurr or current_state and next_state
        :return: next hidden state or probability of transition from given current and next states
        """
        if len(args) == 1:
            current_state = args[0]
            # args = state. Want to get next state
            return self.states[discrete_distribution(self.__transition_probabilities[current_state])]
        if len(args) == 2:
            # args = current state, next state. Want to know probability of transition
            current_state, next_state = args
            return self.__transition_probabilities[current_state][self.states.index(next_state)]

class HmmModel:
    @staticmethod
    def pseudo_length_call_match():
        """
        :return: matrix of length call (case non-zero length)
        """
        residue = [10., 5., 1., 0.5, 10**(-2), 10**(-3), 10**(-4), 10**(-5), 10**(-6), 10**(-7), 10**(-8), 10**(-9),
                   10**(-10), 10**(-11),  10**(-12)]
        length_call_matrix = []
        for i in range(15):
            row = [0] * 15
            row[i] = 100
            for j in range(i):
                row[j] = residue[i - j - 1]
            for j in range(i + 1, 15):
                row[j] = residue[j - i - 1]
            rowFinal = [row[i]/sum(row) for i in range(15)]
            length_call_matrix.append(rowFinal)
        length_call = {'A': length_call_matrix, 'C': length_call_matrix, 'G': length_call_matrix,
                       'T': length_call_matrix}
        return length_call

    @staticmethod
    def pseudo_length_call_insertion():
        """
        :return: vector of probabilities of length call in case insertion
        """
        calls = range(15, 1, -1)
        insertion_calls_probabilities = [calls[i]/float(sum(calls)) for i in range(14)]
        insertion_calls = {'A': insertion_calls_probabilities, 'C': insertion_calls_probabilities,
                           'G': insertion_calls_probabilities, 'T': insertion_calls_probabilities}
        return insertion_calls

    def fill_test_model(self, size):
        """
        Create test model, probabilities of all blocks are equal and extracts from file 'HMM_test.txt'
        :param size:
        :return:
        """
        length_call_test = self.pseudo_length_call_match()
        length_call_insertion_test = self.pseudo_length_call_insertion()
        transition_probabilities = {}
        for lines in open('/home/andina/PycharmProjects/BioInf/Diploma/HMM_test.txt'):
            information = re.split('\t', lines.rstrip('\n'))
            if information[0] == '-:':
                base_call_probabilities = [float(information[i]) for i in range(1, len(information))]
            elif information[0] == 'Begin':
                 transition_probabilities['Begin'] = [float(information[i]) for i in range(1, len(information))]
            elif information[0] == 'Match':
                transition_probabilities['Match'] = [float(information[i]) for i in range(1, len(information))]
            elif information[0] == 'Insertion':
                transition_probabilities['Insertion'] = [float(information[i]) for i in range(1, len(information))]
            elif information[0] == 'Deletion':
                transition_probabilities['Deletion'] = [float(information[i]) for i in range(1, len(information))]
        base_call = {'-': base_call_probabilities}
        element = HmmRecord(base_call, length_call_test, length_call_insertion_test, transition_probabilities)
        self.HMM = [element]*size
        return 0

    def __init__(self, *args):
        self.__modelLength = 20
        self.states = ['Match', 'Deletion', 'Insertion', 'End']
        # self.initial_probabilities = []
        self.HMM = []
        if len(args) == 0:
            # create test model
            print 'Create test model'
            self.fill_test_model(self.__modelLength)
            # later you have to add some code here...

def create_sequence(model, max_size, reference):
    # max_size == len(reference)? Nooooo.
    """
    Model sequence path with given reference
    :param model: HMM model
    :param max_size: max size of sequence. If we don't achieve End until achievement max_size, stop modeling
    :return: path of states and homopolymers
    """
    state_path = []
    sequence = []
    # get initial state, set model size to 0, reference position to 0
    current_state_number = 0
    current_reference_number = 0
    # current_state = model.states[discrete_distribution(model.initial_probabilities)]
    current_state = 'Begin'
    state_path.append([current_state_number, current_state])

    # count for number of insertions
    number_insertions = 0
    """
    if current_state == 'Match':
        current_HP = model.HMM[0].emission(reference[0], 'Match')
        current_reference_number += 1
    if current_state == 'Insertion':
        current_HP = model.HMM[0].emission(reference[0], 'Insertion')
        number_insertions += 1
    if current_state == 'Deletion':
        current_HP = homopolymer()
        current_reference_number += 1
    sequence.append(current_HP)
    """
    while current_state_number != max_size:
        # get next state, while length of model allow us do it
        next_state = model.HMM[current_state_number].transition(current_state)
        if number_insertions == 2:
            while next_state == 'Insertion':
                print 'WTF TOO MANY INSERTIONS'
                next_state = model.HMM[current_state_number].transition(current_state)
        if next_state == 'Match':
            current_HP = model.HMM[current_state_number].emission(reference[current_reference_number], 'Match')
            current_reference_number += 1
            current_state_number += 1
            number_insertions = 0
        elif next_state == 'Insertion':
            current_HP = model.HMM[current_state_number].emission(reference[current_reference_number], 'Insertion')
            number_insertions += 1
        elif next_state == 'Deletion':
            current_HP = homopolymer()
            current_reference_number += 1
            current_state_number += 1
            number_insertions = 0
        else:
            print 'hmm.py, 242 line, error!'
            exit(1)
        sequence.append(current_HP)
        state_path.append([current_state_number, next_state])
        if next_state == 'End' or current_reference_number == len(reference):
            break
        current_state = next_state
    return state_path, sequence


def nucleotide_to_homopolymer(sequence):
    """
    Create homopolymer sequence from nucleotide sequence
    :param sequence: nucleotide sequence
    :return: list of homopolymer
    """
    result = []
    base = sequence[0]
    length = 1
    for i in range(1, len(sequence)):
        if sequence[i] == base:
            length += 1
        else:
            result.append(homopolymer(base, length))
            base = sequence[i]
            length = 1
    result.append(homopolymer(base, length))
    return result

def homopolymer_to_nucleotide(sequence):
    """
    Create nucleotide sequence from homopolymer sequence
    :param sequence: list with homopolymers
    :return: string with nucleotides
    """
    result = ''
    for item in sequence:
        if item.base == '-':
            continue
        else:
            hp = item.base * item.length
            result += hp
    return result
