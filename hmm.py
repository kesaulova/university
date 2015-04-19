import numpy
import re
import addmath

discrete_distribution = addmath.discrete_distribution
eln = addmath.eln
exp = addmath.eexp
log_product = addmath.log_product
iter_plog = addmath.iter_plog
eexp = addmath.eexp

class homopolymer:
    """
    homopolymer = (base, length).
    At example AAAA = ('A', 4)
    """
    def __init__(self, base='-', length=0):
        if length == 0:
            self.base = '-'
            self.length = length
        else:
            self.base = base
            self.length = length


class HmmRecord:
    """
    Correspond to each block in model. Contains:
     1) base call for insertion
     2) length call for Match
     3) length call for insertion
     4) transition probabilities.
     In case null-block, block contain only transition probabilities = initial probabilities, but other parameters
     have no sense.
     All probabilities is log-probabilities

     !!! Here length call doesn't depend from base!
    """
    @staticmethod
    def get_eln(item):
        """
        Take a log from every element in array
        :param item: array (one or more dimensional)
        :return: ndarray with same shape, with log-elements
        """
        nd_item = numpy.array(item).copy()
        for x in numpy.nditer(nd_item, op_flags=['readwrite']):
            x[...] = eln(x)
        return nd_item

    @staticmethod
    def get_eexp(item):
        """
        Take a eexp from every element in array
        :return: ndarray with same shape, with exp-elements
        """
        nd_item = numpy.array(item).copy()
        for x in numpy.nditer(nd_item, op_flags=['readwrite']):
            x[...] = eexp(x)
        return nd_item

    def __init__(self, base_call, length_call, length_call_insert, transition_probabilities):
        """
        Constructor
        :param base_call: dictionary, three key: Deletion (value - probability of remove reference base),
        Insertion(value - probability of insert read base), Match (value - 1?)
        :param length_call_insert: dictionary, key - bases, values - length of gomopolymers of such bases when
        insertions occur
        :param length_call: matrix with size max_hp_size * max_hp_size
        :param transition_probabilities: dictionary with trans.prob. for every state
        :return:
        """
        self.bases = ['A', 'C', 'G', 'T']
        self.states = ['Match', 'Deletion', 'Insertion', 'Begin', 'End']

        if sum(base_call) > 0:
            self.__insert_base_call = self.get_eln(base_call)
        else:
            self.__insert_base_call = base_call

        if sum(length_call_insert) > 0:
            self.__length_call_insert = self.get_eln(length_call_insert)
        else:
            self.__length_call_insert = length_call_insert

        if sum(length_call[0]) > 0:
            self.__length_call_match = self.get_eln(length_call)
        else:
            self.__length_call_match = length_call

        if sum(transition_probabilities[0]) > 0:
            self.__transition_probabilities = self.get_eln(transition_probabilities)
        else:
            self.__transition_probabilities = transition_probabilities

        # self__length_call_match = {'A': length_call['A'], 'C': length_call['C'], 'G': length_call['G'],
        #                                     'T': length_call['T']}
        # self.__length_call_insert = {'A': eln(length_call_insert['A']), 'C': eln(length_call_insert['C']),
        #                                         'G': eln(length_call_insert['G']), 'T': eln(length_call_insert['T'])}

    def __emiss_not_viterbi(self, h, state):
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
            # [h.length + 1] - count from 0
            # bins = numpy.cumsum(self__length_call_match[h.base][h.length - 1])
            tmp = self.get_eexp(self.__length_call_match[h.length - 1])
            bins = numpy.cumsum(tmp)
            length = 0
            while length == 0:
                length = numpy.digitize(numpy.random.random_sample(1), bins)[0]
            return homopolymer(h.base, length)
        if state == 'Insertion':
            tmp = self.get_eexp(self.__insert_base_call)
            bins = numpy.cumsum(tmp)
            base = self.bases[numpy.digitize(numpy.random.random_sample(1), bins)]
            length = 0
            while length == 0:
                # bins = numpy.cumsum(self.__length_call_insert[base])
                tmp = self.get_eexp(tmp)
                bins = numpy.cumsum(self.__length_call_insert[self.bases.index(base)])
                # '+ 1' - we count from 0
                length = numpy.digitize(numpy.random.random_sample(1), bins)[0] + 1
            return homopolymer(base, length)

    def __emiss_viterbi(self, h, g, state):
        """
        :param h: input homopolymer
        :param g: output homopolymer
        :return: probability that state 'state' emits g with given h
        """
        if state == 'Begin':
            return 1
        if state == 'Deletion':
            if g.base != '-' or h.base == '-':   # if it is not deletion
                return 0
            return 1
        if state == 'Match':
            if h.base != g.base or h.base == '-' or g.base == '-':    # if it isn't match
                return 0
            # [g.length - 1] - count from 0
            # return self__length_call_match[h.base][h.length][g.length - 1]
            return self.__length_call_match[h.length - 1][g.length - 1]
        if state == 'Insertion':
            if h.base != '-' or g.base == '-':
                return 0
            # [g.length - 1] - count from 0
            # return (self.__insert_base_call[self.bases.index(g.base)] *
            #      self.__length_call_insert[g.base][g.length - 1])
            return log_product(self.__insert_base_call[self.bases.index(g.base)],
                  self.__length_call_insert[g.length - 1])

    def emission(self, *args):
        """
        return emission probabilty
        :param args: in case 2 args - homopolymer and state, returns output homopolymer, accordingly to emission probs.
        In case 3 srgs - two HP and state, return probability of observing such output and input
        :return: homopolymer (case 2 args)or probability(case 3 args)
        """
        if len(args) == 2:
            # args = g, state. Need to know homopolymer output
            return self.__emiss_not_viterbi(*args)
        if len(args) == 3:
            # args = g, h, state. Want to know probability
            return self.__emiss_viterbi(*args)

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
            return self.states[discrete_distribution(self.get_eexp(self.__transition_probabilities[self.states.index(current_state)]))]
        elif len(args) == 2:
            # args = current state, next state. Want to know probability of transition
            current_state, next_state = args
            # transitions probabilities is dictionary
            # print 'curr: ', current_state, "next: ", next_state
            return self.__transition_probabilities[self.states.index(current_state)][self.states.index(next_state)]
        else:
            print "Error with length of arguments if transition, length = ", len(args)
            exit(1)


class HmmModel:
    """
    Realize HMM model. If there is no input in creation an instance, then creates model himself,
    using data from HMM_test.txt.
    Model contains from blocks, number of blocks have predefine size. Each block - HMM record.
    ! Don't have method for fill Hmm Model in not-testing case
    """
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
        #  length_call = {'A': length_call_matrix, 'C': length_call_matrix, 'G': length_call_matrix,
        #                'T': length_call_matrix}
        return length_call_matrix

    @staticmethod
    def pseudo_length_call_insert():
        """
        :return: vector of probabilities of length call in case insertion
        """
        calls = range(15, 1, -1)
        insertion_calls_probabilities = [calls[i]/float(sum(calls)) for i in range(14)]
        # insertion_calls = {'A': insertion_calls_probabilities, 'C': insertion_calls_probabilities,
        #                  'G': insertion_calls_probabilities, 'T': insertion_calls_probabilities}
        insertion_calls = insertion_calls_probabilities
        return insertion_calls

    def fill_test_model(self, size):
        """
        Create test model, probabilities of all blocks are equal and extracts from file 'HMM_test.txt'
        :param size: number of blocks in model
        :return:
        """
        length_call_test = self.pseudo_length_call_match()
        length_call_insert_test = self.pseudo_length_call_insert()
        transition_probabilities = numpy.zeros(shape=[len(self.states), len(self.states)], dtype=float)
        for lines in open('./HMM_test.txt'):
            # print re.split('\t', lines.rstrip('\n'))
            information = re.split('\t', lines.rstrip('\n'))
            if information[0] == '-:':
                base_call = [float(information[i]) for i in range(1, len(information))]
            elif information[0] == 'Begin':
                transition_probabilities[self.states.index('Begin')] = [float(information[i]) for i in range(1, len(information))]
            elif information[0] == 'Match':
                transition_probabilities[self.states.index('Match')] = [float(information[i]) for i in range(1, len(information))]
            elif information[0] == 'Insertion':
                transition_probabilities[self.states.index('Insertion')] = [float(information[i]) for i in range(1, len(information))]
            elif information[0] == 'Deletion':
                transition_probabilities[self.states.index('Deletion')] = [float(information[i]) for i in range(1, len(information))]
            elif information[0] == 'End':
                transition_probabilities[self.states.index('End')] = [float(information[i]) for i in range(1, len(information))]
        element = HmmRecord(base_call, length_call_test, length_call_insert_test, transition_probabilities)
        self.HMM = [element]*size
        return 0

    def __init__(self, *args):
        """
        Sequence of arguments: base_call, len_call_match, len_call_ins, trans_prob
        """
        self.__modelLength = 400
        self.states = ['Match', 'Deletion', 'Insertion', 'Begin', 'End']
        self.HMM = []
        if len(args) == 0:
            # create test model
            print 'Create test model'
            self.fill_test_model(self.__modelLength)
        else:
            assert (len(args) == 4), "Length of arguments for HMM model doesn't equal 4!"
            base_call, len_call_match, len_call_ins, trans_prob = args
            if isinstance(trans_prob, list):
                self.HMM = []
                for i in range(len(trans_prob)):
                    element = HmmRecord(base_call[i], len_call_match[i], len_call_ins[i], trans_prob[i])
                    self.HMM.append(element)
            else:
                element = HmmRecord(base_call, len_call_match, len_call_ins, trans_prob)
                self.HMM = [element]*400


def create_sequence(model, max_size, reference):
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
    while current_state_number != max_size:
        # get next state, while length of model allow us do it
        next_state = model.HMM[current_state_number].transition(current_state)
        if number_insertions == 2:
            while next_state == 'Insertion':
                print 'WTF TOO MANY INSERTIONS'
                next_state = model.HMM[current_state_number].transition(current_state)
        if next_state == 'Match':
            current_hp = model.HMM[current_state_number].emission(reference[current_reference_number], 'Match')
            current_reference_number += 1
            current_state_number += 1
            number_insertions = 0
        elif next_state == 'Insertion':
            current_hp = model.HMM[current_state_number].emission(reference[current_reference_number], 'Insertion')
            number_insertions += 1
        elif next_state == 'Deletion':
            current_hp = homopolymer()
            current_reference_number += 1
            current_state_number += 1
            number_insertions = 0
        else:
            print 'hmm.py, 242 line, error!'
            exit(1)
        sequence.append(current_hp)
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
