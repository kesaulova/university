from math import log, exp
import numpy
import re
import addmath
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
        #if state == 'Begin'
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
                self.initial_probabilities = [float(information[i]) for i in range(1, len(information))]
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
        self.initial_probabilities = []
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
    current_state = model.states[discrete_distribution(model.initial_probabilities)]
    state_path.append([current_state_number, current_state])
    # count for number of insertions
    number_insertions = 0
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
        if next_state == 'Insertion':
            current_HP = model.HMM[current_state_number].emission(reference[current_reference_number], 'Insertion')
            number_insertions += 1
        if next_state == 'Deletion':
            current_HP = homopolymer()
            current_reference_number += 1
            current_state_number += 1
            number_insertions = 0
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

def process_read(read):
    """
    Create string were at each positions is length of lengthest homopolymer, ended at these position
    :param read: nucleotide sequence
    :return: list with numbers, indicated length of the longest homopolymer, ended at these position
    """
    result = [1]*len(read)
    base = read[0]
    for i in range(1, len(read)):
        if read[i] == base:
            result[i] = result[i - 1] + 1
        else:
            base = read[i]
            result[i] = 1
    return result

def parse_backtracking(index_string):
    """
    Parse string of indexes in list of integers
    :param index_string: string whith numbers, separated by \t
    :return: List with four elements
    """
    if index_string == 'Begin':
        return [0, 0, 0, 0]
    else:
        result = [int(item) for item in re.split(' ', index_string)]
        return result

def length_first_hp(read):
    result = 1
    base = read[1]
    for i in range(2, len(read)):
        if read[i] == base:
            result += 1
        else:
            break
    return result


def viterbi_initialize(model, reference, read, k_max, viterbi_probability,  viterbi_backtracking):
    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion'}
    # V(0,j,0,M) = V(0,j,0,I) = 0, V(0,j,0,D) != 0
    print viterbi_backtracking.shape
    print viterbi_probability.shape
    viterbi_probability[0][0][0][0] = 0
    viterbi_backtracking[0][0][0][0] = 'Impossible'
    viterbi_probability[0][0][0][1] = 0
    viterbi_backtracking[0][0][0][0] = 'Impossible'
    viterbi_probability[0][0][0][2] = 0
    viterbi_backtracking[0][0][0][0] = 'Impossible'
    viterbi_probability[0][0][0][3] = 0
    viterbi_backtracking[0][0][0][0] = 'Impossible'
    # V(0,0,k,pi) = 0
    for k in range(max(k_max)):
        for state in states:
            viterbi_backtracking[0][0][k][state] = 'Impossible'
            viterbi_probability[0][0][k][state] = 0
    for j in range(1, len(reference)):
        viterbi_probability[0][j][0][0] = 0
        viterbi_backtracking[0][j][0][0] = 'Impossible'
        viterbi_probability[0][j][0][2] = 0
        viterbi_backtracking[0][j][0][2] = 'Impossible'
        if j == 1:
            viterbi_probability[0][j][0][1] = eln(model.initial_probabilities[1])
            # print 0, j, 0, states[1], viterbi_probability[0][j][0][1]
            viterbi_backtracking[0][j][0][1] = 'Begin'
        else:
            viterbi_probability[0][j][0][1] = log_product(eln(model.HMM[j].transition('Deletion', 'Deletion')),
                                                          viterbi_probability[0][j - 1][0][1])
            # print 0, j, 0, states[1], viterbi_probability[0][j][0][1]
            viterbi_backtracking[0][j][0][1] = str(0) + ' ' + str(j - 1) + ' ' + str(0) + ' ' + str(1)
    # V(i,0,k,M) = V(i,0,k,D) = 0
    for i in range(1, len(read)):
        # k count from 1 - it is length of HP in sequence
        for k in range(1, k_max[i] + 1):
            viterbi_probability[i][0][k][0] = 0     # V(i,0,k,Match)
            viterbi_backtracking[i][0][k][0] = 'Impossible'
            viterbi_probability[i][0][k][1] = 0    # V(i,0,k,Deletion)
            viterbi_backtracking[i][0][k][0] = 'Impossible'
            # Count V(i,0,k,Insertion). Have to find max among V(i - k, 0, 1:k_max[i - k], Insertion)
            # 1:k_max[i - k], because there is no deletions
            # First, initialize V(i,0,k,I) when i < k_max[(len(first homopolymer)], because in this case transition
            # probabilities is initial probabilities
            if i <= length_first_hp(read):
                for k_prev in range(1, k_max[i] + 1):
                    viterbi_probability[i][0][k_prev][2] = eln(model.initial_probabilities[2])
                    viterbi_backtracking[i][0][k_prev][2] = 'Begin'
                    #print i, 0, k_prev, states[2], viterbi_backtracking[i][0][k_prev][2]
            else:
                max_prob = float("-inf")
                number = [0, 0, 0, 0]
                current_hp = homopolymer(read[i], k)
                for k_prev in range(1, k_max[i - k] + 1):
                    # Have to find max among V(i - k, 0, 1:k_max[i - k], Insertion)
                    if viterbi_probability[i - k][0][k_prev][2] > max_prob:
                        max_prob = viterbi_probability[i - k][0][k_prev][2]
                        number = [i - k, 0, k_prev, 2]
                    # count probability
                trans_prob = eln(model.HMM[0].transition('Insertion', 'Insertion'))
                emiss_prob = eln(model.HMM[0].emission(homopolymer(), current_hp, 'Insertion'))
                viterbi_probability[i][0][k][2] = iter_plog([max_prob, trans_prob, emiss_prob])
                viterbi_backtracking[i][0][k][2] = str(number[0]) + ' ' + str(number[1]) + ' ' + str(number[2]) + \
                                                       ' ' + str(number[3])
    return viterbi_probability, viterbi_backtracking


def viterbiPath(read, reference, model):
    """
    :param read: nucleotide sequence
    :param reference: nucleotide sequence. We transform it to homopolymer sequence
    :return: most probable path of hidden state and its probability
    """
    #reference = nucleotide_to_homopolymer(reference)    # make homopolymer sequence
    k_max = [0]
    k_max.extend(process_read(read))
    print k_max
    read = ' ' + read
    temp = [homopolymer()]
    temp.extend(reference)
    reference = temp
    max_k_value = max(k_max) + 1
    # states = {'Match': 0, 'Deletion': 1, 'Insertion': 2}#, 'End': 3}
    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion'}  # , 3: 'End'}
    viterbi_probability = (-1)*numpy.ones(shape=[len(read), len(reference), max_k_value, 4], dtype=float)
    viterbi_backtracking = (-1)*numpy.ones(shape=[len(read), len(reference), max_k_value, 4], dtype=basestring)
    # initialize model, process i = 0, j = 0 cases
    viterbi_probability, viterbi_backtracking = viterbi_initialize(model, reference, read, k_max, viterbi_probability,
                                                                   viterbi_backtracking)
    for i in range(1, len(read)):
        for j in range(1, len(reference)):
            for k in range(1, k_max[i] + 1):
                # process 'Match' case: V(i, j, k, M) = max(k',pi')(V(i - k, j - 1, k', pi')*p(M|pi')*emission
                if read[i] != reference[j].base:    # in Match bases should be the same, not the different
                    viterbi_probability[i][j][k][0] = 0
                    print i, j, k, states[0], 'Prob: ', viterbi_probability[i][j][k][0], read[i], reference[j].base
                else:
                    max_prob = float("-inf")
                    prev_index = [0]*4
                    # previous homopolymer can have length from 1 to k_max[prev] + 1
                    # we want to find most probable path until this moment
                    for k_prev in range(1, k_max[i - k] + 1):
                        emiss_prob = eln(model.HMM[j].emission(reference[j], homopolymer(read[i], k), 'Match'))
                        for state in states:    # from what state we come
                            trans_prob = eln(model.HMM[j - 1].transition(states[prev_index[3]], 'Match'))
                            print 'Previous', i - k, j - 1, k_prev, states[state], \
                                viterbi_probability[i - k][j - 1][k_prev][state], i, j, k, k_prev
                            value = log_product(viterbi_probability[i - k][j - 1][k_prev][state], trans_prob)
                            # our matrix fill with zeros, but logarithm from probability is negative. Should remember
                            # about that when trying to find max
                            if (max_prob == 0 or value == 0) and \
                                            max_prob != float("-inf"):
                                if abs(viterbi_probability[i - k][j - 1][k_prev][state]) > abs(max_prob):
                                    max_prob = viterbi_probability[i - k][j - 1][k_prev][state]
                                    prev_index = [i - k, j - 1, k_prev, state]
                            elif viterbi_probability[i - k][j - 1][k_prev][state] > max_prob:
                                max_prob = viterbi_probability[i - k][j - 1][k_prev][state]
                                prev_index = [i - k, j - 1, k_prev, state]
                    # check if we come from begin
                    if i <= length_first_hp(read) and j == 1:
                        trans_prob = eln(model.initial_probabilities[0])
                        max_prob = 1
                        viterbi_backtracking[i][j][k][0] = 'Begin'
                    else:
                        trans_prob = eln(model.HMM[j - 1].transition(states[prev_index[3]], 'Match'))
                        viterbi_backtracking[i][j][k][0] = str(prev_index[0]) + ' ' + str(prev_index[1]) + ' ' + \
                                                       str(prev_index[2]) + '\t' + str(prev_index[3])
                    emiss_prob = eln(model.HMM[j].emission(reference[j], homopolymer(read[i], k), 'Match'))
                    viterbi_probability[i][j][k][0] = iter_plog([max_prob, trans_prob, emiss_prob])
                    print 'Previous - max', max_prob, prev_index
                    print i, j, k, states[0], 'Prob: ', viterbi_probability[i][j][k][0], read[i], reference[j].base

                # process 'Insertion' case: V(i, j, k, I) = max(k',pi')(V(i - k, j, k', pi')*p(I|pi')*emission
                # almost same, but here j not changed
                max_prob = float("-inf")
                prev_index = [0]*4
                if k_max[i - k] == 0:
                    for state in states:
                        print 'Previous:', i - k, j, 0, states[state], viterbi_probability[i - k][j][0][state]
                        if (max_prob == 0 or viterbi_probability[i - k][j][0][state] == 0) and max_prob != float("-inf"):
                            if abs(viterbi_probability[i - k][j][0][state]) > abs(max_prob):
                                max_prob = viterbi_probability[i - k][j][0][state]
                                prev_index = [i - k, j, 0, state]
                        elif viterbi_probability[i - k][j][0][state] > max_prob:
                            max_prob = viterbi_probability[i - k][j][0][state]
                            prev_index = [i - k, j, 0, state]
                for k_prev in range(1, k_max[i - k] + 1):
                    for state in states:
                        print 'Previous:', i - k, j, k_prev, states[state], viterbi_probability[i - k][j][k_prev][state]
                        if (max_prob == 0 or viterbi_probability[i - k][j][k_prev][state] == 0) and max_prob != float("-inf"):
                            if abs(viterbi_probability[i - k][j][k_prev][state]) > abs(max_prob):
                                max_prob = viterbi_probability[i - k][j][k_prev][state]
                                prev_index = [i - k, j, k_prev, state]
                        elif viterbi_probability[i - k][j][k_prev][state] > max_prob:
                            max_prob = viterbi_probability[i - k][j][k_prev][state]
                            prev_index = [i - k, j, k_prev, state]
                if max_prob == float("-inf"):
                    max_prob = 0
                trans_prob = eln(model.HMM[j].transition(states[prev_index[3]], 'Insertion'))
                emiss_prob = eln(model.HMM[j].emission(homopolymer(), homopolymer(read[i], k), 'Insertion'))
                viterbi_probability[i][j][k][2] = iter_plog([max_prob, trans_prob, emiss_prob])
                print 'Previous - max', max_prob, prev_index
                print i, j, k, states[2], viterbi_probability[i][j][k][2]
                viterbi_backtracking[i][j][k][2] = str(prev_index[0]) + ' ' + str(prev_index[1]) + ' ' + str(prev_index[2]) + \
                                                   '\t' + str(prev_index[3])

                # process 'Deletion' case: V(i, j, k, I) = max(pi')(V(i, j - 1, k', pi')*p(D|pi')*emission of '-'
                max_prob = float("-inf")
                prev_index = []
                for state in states:
                    if (max_prob == 0 or viterbi_probability[i][j - 1][k][state] == 0) and max_prob != float("-inf"):
                        if abs(viterbi_probability[i][j - 1][k][state]) > abs(max_prob):
                            max_prob = viterbi_probability[i][j - 1][k][state]
                            prev_index = [i, j - 1, k, state]
                    elif viterbi_probability[i][j - 1][k][state] > max_prob:
                        max_prob = viterbi_probability[i][j - 1][k][state]
                        prev_index = [i, j - 1, k, state]
                if max_prob == float("-inf"):
                    max_prob = 0
                trans_prob = eln(model.HMM[j - 1].transition(states[prev_index[3]], 'Deletion'))
                #emiss_prob = eln(model.HMM[j].emission(reference[j], homopolymer(), 'Deletion'))
                emiss_prob = 1
                viterbi_probability[i][j][k][1] = iter_plog([max_prob, trans_prob, emiss_prob])
                print 'Previous: - max', max_prob, prev_index
                print i, j, k, states[1], viterbi_probability[i][j][k][1]
                viterbi_backtracking[i][j][k][1] = str(prev_index[0]) + ' ' + str(prev_index[1]) + ' ' + str(prev_index[2]) + \
                                                   '\t' + str(prev_index[3])

    return viterbi_probability, viterbi_backtracking
