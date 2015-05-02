import hmm
import numpy
import addmath
import math
from scipy.stats import laplace, lognorm
from scipy.integrate import quad
import re
from inspect import currentframe, getframeinfo

eln = addmath.eln
eexp = addmath.eexp
log_product = addmath.log_product
log_sum = addmath.log_sum
iter_plog = addmath.iter_plog
iter_slog = addmath.iter_slog
homopolymer = hmm.homopolymer
nucl_to_hp = hmm.nucleotide_to_homopolymer
by_iter_slog = addmath.by_iter_slog

def length_last_hp(read):
    result = 1
    base = read[len(read) - 1]
    for i in xrange(len(read) - 2, -1, -1):
        if read[i] == base:
            result += 1
        else:
            break
    return result


def len_max_hp_end(sequence):
    """
    Create list were at each positions is length of lengthiest hmm.homopolymer, ended at these position
    :param sequence: nucleotide sequence
    :return: list with numbers, indicated length of the longest hmm.homopolymer, ended at these position
    """
    result = [1]*len(sequence)
    base = sequence[0]
    for i in xrange(1, len(sequence)):
        if sequence[i] == base:
            result[i] = result[i - 1] + 1
        else:
            base = sequence[i]
            result[i] = 1
    return result


def len_max_hp_start(sequence):
    """
    Create list were at each positions is length of lengthiest hmm.homopolymer, starting at these position
    :param sequence: nucleotide sequence
    :return: list with numbers, indicated length of the longest hmm.homopolymer, ended at these position
    """
    result = [1]*len(sequence)
    base = sequence[len(sequence) - 1]
    for i in xrange(len(sequence) - 2, -1, -1):
        if sequence[i] == base:
            result[i] = result[i + 1] + 1
        else:
            base = sequence[i]
            result[i] = 1
    return result


# I count F and B from 1! Read and reference. In HMM 0 block carries information about initial probability distribution
def count_forward(read_tmp, reference_tmp, model):
    """
    Count forward variable F(i,j,k,l,pi) - is the probability summarizing over all possible alignments ending at
    the hidden state pi between the prefixes read[1:i] and reference[1:j] in which a k bp homopolymer at the
    read position i is aligned to a l bp homopolymer in the reference position j.
    :param read_tmp: read, nucleotide sequence
    :param reference_tmp: reference, nucleotide sequence
    :param model: HMM model
    :return: array of forward variables
    """
    # first create homopolymer sequence from reference
    reference = [homopolymer()] + nucl_to_hp(reference_tmp)
    read = ' ' + read_tmp
    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion', 3: 'Begin', 4: 'End'}
    states_without_ins = {0: 'Match', 1: 'Deletion', 3: 'Begin', 4: 'End'}
    states_without_match = {1: 'Deletion', 2: 'Insertion', 3: 'Begin', 4: 'End'}

    max_hp_read = [0] + len_max_hp_end(read_tmp) + [0]


    # WHY WITHOUT 0
    len_hp_ref = len_max_hp_end(reference_tmp)   # information about the length of HP in reference


    forward = float("-inf")*numpy.ones(shape=[len(read), len(reference), max(max_hp_read) + 1,
                                              max(len_hp_ref) + 1, len(states)], dtype=float)
    forward_position = float("-inf")*numpy.ones(shape=[len(read), len(reference), len(states)], dtype=float)
    forward_position[0][0][3] = 0
    print "read: ", 0, "len: ", 0, "ref: ", 0, "len: ", 0, states[3], forward_position[0][0][3]


    def process_match(i, j, k, hp_input, hp_output):
        """
        Count F(i, j, k, l, Match).
        We must check that there is no transitions ins-match, if read[i] == read[i - k].
        :param i: read position
        :param j: reference position
        :param k: length of HP, ended at position i
        :param l: length of HP, ended at position j
        :param hp_input: HP in ref
        :param hp_output: HP in read
        :return:
        """

        if read[i] == read[i - k]:
            possible_prev_states = states_without_ins
        else:
            possible_prev_states = states

        result = float("-inf")
        for prev_state in possible_prev_states:
            transition = model.HMM[j - 1].transition(states[prev_state], 'Match')
            emission = model.HMM[j].emission(hp_input, hp_output, 'Match')
            print transition, emission, forward_position[i - k, j - 1, prev_state]
            result = log_sum(result, iter_plog([forward_position[i - k, j - 1, prev_state], transition, emission]))
        forward[i, j, k, reference[j].length, 0] = result
        print "read: ", i, "len: ", k, "ref: ", j, "len: ", reference[j].length, states[0], forward[i, j, k, reference[j].length, 0]
        # print i, j, k, l, states[0], forward[i, j, k, reference[j].length, 0]
        return 0

    def process_insertion(i, j, k, hp_output):
        """
        Count F(i, j, k, l, Insertion).
        We must check that there is no transitions match-ins, if read[i] == read[i - k].
        :param i: read position
        :param j: reference position
        :param k: length of HP, ended at position i
        :param l: length of HP, ended at position j, = 0
        :param hp_output: HP in read
        :return:
        """
        if read[i] == read[i - k]:
            possible_prev_states = states_without_match
        else:
            possible_prev_states = states
        result = float("-inf")
        for prev_state in possible_prev_states:
            transition = model.HMM[j].transition(states[prev_state], 'Insertion')
            emission = model.HMM[j].emission(homopolymer(), hp_output, 'Insertion')
            result = log_sum(result, iter_plog([forward_position[i - k, j, prev_state], transition, emission]))
        forward[i, j, k, 0, 2] = result
        print "read: ", i, "len: ", k, "ref: ", j, "len: ", 0, states[2], forward[i, j, k, 0, 2]
        # print i, j, k, 0, states[2], forward[i, j, k, 0, 2]
        return 0

    def process_deletion(i, j):
        """
        Count F(i, j, k, l, Deletion).
        :param i: read position
        :param j: reference position
        :param k: length of HP, ended at position i, = 0
        :param l: length of HP, ended at position j
        :param hp_input: HP in reference
        :return:
        """
        result = float("-inf")
        for prev_state in states:
            transition = model.HMM[j - 1].transition(states[prev_state], 'Deletion')
            result = log_sum(result, iter_plog([forward_position[i, j - 1, prev_state], transition]))
        forward[i, j, 0, reference[j].length, 1] = result
        print "read: ", i, "len: ", 0, "ref: ", j, "len: ", reference[j].length, states[1], forward[i, j, 0, reference[j].length, 1]
        # print i, j, k, reference[j].length, states[1], forward[i, j, k, reference[j].length, 1]
        return 0

    # start from 0, because there can be insertions and deletions at the begining
    for i in xrange(len(read)):

        for j in xrange(len(reference)):

            if i == j == 0:     # Can't align two empty hp
                continue

            print "\nReference: ",  hmm.hp_print(reference[:j+1]), "Residue: ", hmm.hp_print(reference[j+1:])
            print "Read: ",  read[:i + 1], "Residue: ", read[i + 1:], '\n'

            for k in range(max_hp_read[i] + 1):

                for l in range(2):

                    if l == 0 and k == 0:
                        continue
                    elif l == 0:
                        process_insertion(i, j, k, homopolymer(read[i], k))
                    elif k == 0:
                        process_deletion(i, j)
                    elif read[i] == reference[j].base:
                        process_match(i, j, k, reference[j], homopolymer(read[i], k))

            # fill F(i,j,pi)
            for state in states:
                forward_position[i][j][state] = by_iter_slog(numpy.nditer(forward[i, j, :, :, state]))

    check = by_iter_slog(numpy.nditer(forward_position[len(read_tmp), len(reference) - 1, :]))
    print forward_position[len(read_tmp), len(reference) - 1, :]

    return forward, check


def count_backward(read_tmp, reference_tmp, model):
    """
    Count backward variable B(i, j, k, l, pi) - probability of all possible alignments of the
    suffixes r[i + 1: n] and t[j + 1: m], starting at the hidden state pi , where a k bp homopolymer at the
    read position i is aligned to a l bp homopolymer at the reference position j.
    We can count B(i, j, ..) and then fill all variables.
    :param read_tmp:
    :param reference_tmp:
    :param model:
    :return: ndarray
    """
    read = ' ' + read_tmp
    reference = [homopolymer()] + nucl_to_hp(reference_tmp)
    len_ref = len(reference) - 1  # true length
    len_read = len(read) - 1  # true length
    # position and create len_sequence to remember true length
    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion', 3: 'Begin', 4: 'End'}
    state_index = {'Match': 0, 'Deletion': 1, 'Insertion': 2, 'Begin': 3, 'End': 4}
    max_hp_read_s = [0] + len_max_hp_start(read_tmp)  # because read start from ' '

    # information about the lengthiest HP, ended at this position
    max_hp_read_e = [0] + len_max_hp_end(read_tmp)  # because read start from ' '
    max_hp_ref_e = [0] + len_max_hp_end(reference_tmp)  # because reference start from ' '

    backward = float("-inf")*numpy.ones(shape=[len_read + 1, len_ref + 1, max(max_hp_read_e) + 1,
                                               max(max_hp_ref_e) + 1,  len(states)], dtype=float)
    # initialize
    # B(i, m, 0, 1, Deletion)
    backward[len_read, len_ref, 0, reference[len_ref].length, state_index['Deletion']] = 0
    print "read: ", len_read, "len: ", 0, "ref: ", len_ref, "len: ", reference[len_ref].length, states[1], \
                        backward[len_read, len_ref, 0, reference[len_ref].length, state_index['Deletion']]
    # B(n, m, k, 0, Insertion)
    emiss = [model.HMM[len_ref].emission(homopolymer(), homopolymer(read[len_read], t), 'Insertion')
             for t in xrange(1, length_last_hp(read) + 1)]
    # backward[len_read, len_ref, 1: length_last_hp(read) + 1, 0, state_index['Insertion']] = emiss
    backward[len_read, len_ref, 1: length_last_hp(read) + 1, 0, state_index['Insertion']] = 0
    for k in range(1, length_last_hp(read) + 1):
        print "read: ", len_read, "len: ", k, "ref: ", len_ref, "len: ", 0, states[2], \
                            backward[len_read, len_ref, k, 0, state_index['Insertion']]

    # B(n, m, k, 1, Match)
    emiss = [model.HMM[len_ref].emission(reference[len_ref], homopolymer(read[len_read], t), 'Match')
             for t in xrange(1, length_last_hp(read) + 1)]
    # backward[len_read, len_ref, 1: length_last_hp(read) + 1, reference[len_ref].length, state_index['Match']] = emiss
    backward[len_read, len_ref, 1: length_last_hp(read) + 1, reference[len_ref].length, state_index['Match']] = 0
    for k in range(1, length_last_hp(read) + 1):
        print "read: ", len_read, "len: ", k, "ref: ", len_ref, "len: ", reference[len_ref].length, states[0], \
                            backward[len_read, len_ref, k, reference[len_ref].length, state_index['Match']]

    def process_match(i, j):
        """
        Count multiply of probability of emission and corresponding backward variable.
        :param i: read position
        :param j: reference position
        :return:
        """
        hp_len = max_hp_read_s[i + 1]
        read_base = read[i + 1]
        hp_input = reference[j + 1]
        st_index = state_index['Match']
        emiss = [model.HMM[j + 1].emission(hp_input, homopolymer(read_base, tt), 'Match') for tt in xrange(1, hp_len + 1)]
        bck = numpy.array([backward[i + tt, j + 1, tt, reference[j + 1].length, st_index] for tt in xrange(1, hp_len + 1)])
        result = emiss + bck
        print "Resuuult: ", result, emiss, bck
        if len(result) == 1:
            return result[0]
        else:
            return iter_slog(result)

    def process_deletion(i, j):
        """
        Count multiply of probability of emission and corresponding backward variable. In deletion case, emission
        probability = 1
        :param i: read position
        :param j: reference position
        """
        # if reference[j ].base == '-':
        #     return float("-Inf")
        return backward[i, j + 1, 0, reference[j + 1].length, state_index['Deletion']]

    def process_insertion(i, j):
        """
        Count multiply of probability of emission and corresponding backward variable.
        :param i: read position
        :param j: reference position
        """
        st_index = state_index['Insertion']
        hp_len = max_hp_read_s[i + 1]
        read_base = read[i + 1]
        hp_input = homopolymer()
        emiss = [model.HMM[j].emission(hp_input, homopolymer(read_base, tt), 'Insertion') for tt in xrange(1, hp_len + 1)]
        bck = numpy.array([backward[i + tt, j, tt, 0, st_index] for tt in xrange(1, hp_len + 1)])
        result = emiss + bck
        if len(result) == 1:
            return result[0]
        else:
            return iter_slog(result)

    for i in xrange(len_read, -1, -1):  # read position
        for j in xrange(len_ref, -1, -1):     # reference position
            if j == len_ref and i == len_read:
                continue

            print "\nReference: ",  hmm.hp_print(reference[:j+1]), "Residue: ", hmm.hp_print(reference[j+1:])
            print "Read: ",  read[:i + 1], "Residue: ", read[i + 1:], '\n'


            # First count \sum\limits{i,j} p(beta_(i + di)|alpha_(j + dj), pi')*p(k|l, pi')*B(i + di, j + dj, k, pi').
            # It is probability of suffix.
            # It will be vector of length 3. Then, for each state, we create vector of transition probabilities,
            # element-wise multiply them and then get sum.
            part_two = numpy.array([(-1)*numpy.inf] * len(states))
            if j != len_ref and i != len_read and read[i + 1] == reference[j + 1].base:
                part_two[0] = process_match(i, j)
            if j != len_ref:
                part_two[1] = process_deletion(i, j)
            if i != len_read:
                part_two[2] = process_insertion(i, j)

            print "Part two: ", part_two

            if i != 0 or j != 0:
                # Count B(i, j, k, l, Match)
                if read[i] == reference[j].base:
                    trans_prob = [model.HMM[j].transition('Match', states[k]) for k in xrange(len(states))]
                    if i != len_read and read[i] == read[i + 1]:
                        trans_prob[2] = float("-inf")
                    value = iter_slog(trans_prob + part_two)
                    for k in xrange(1, max_hp_read_e[i] + 1):
                        backward[i, j, k, reference[j].length, state_index['Match']] = value
                        print "read: ", i, "len: ", k, "ref: ", j, "len: ", reference[j].length, states[0], \
                            backward[i, j, k, reference[j].length, state_index['Match']]

                # Count B(i, j, k, l, Deletion)
                trans_prob = [model.HMM[j].transition('Deletion', states[k]) for k in xrange(len(states))]
                value = iter_slog(trans_prob + part_two)
                backward[i, j, 0,  reference[j].length, state_index['Deletion']] = value

                print "read: ", i, "len: ", 0, "ref: ", j, "len: ", reference[j].length, states[1], \
                    backward[i, j, 0,  reference[j].length, state_index['Deletion']]

                # Count B(i, j, k, l, Insertion)
                trans_prob = [model.HMM[j].transition('Insertion', states[k]) for k in xrange(len(states))]
                if i != len_read and read[i] == read[i + 1]:
                    trans_prob[0] = float("-inf")
                value = iter_slog(trans_prob + part_two)
                for k in xrange(1, max_hp_read_e[i] + 1):
                    backward[i, j, k, 0, state_index['Insertion']] = value
                    print "read: ", i, "len: ", k, "ref: ", j, "len: ", 0, states[2], backward[i, j, k, 0, state_index['Insertion']]

            else:   # i == 0 and j == 0:     # Begin case!
                trans_prob = [model.HMM[0].transition('Begin', states[k]) for k in xrange(len(states))]
                backward[0, 0, 0, 0, 3] = iter_slog(trans_prob + part_two)
                print "read: ", i, "len: ", 0, "ref: ", j, "len: ", 0, states[3], backward[i, j, 0,  0, 3]
                print trans_prob
                print part_two
    return backward, backward[0, 0, 0, 0, 3]


read =      "TTCTTTGCCGCACTGGCCCGCGCCAATATCAACATTGTCGCCATTGCTCAGGGATCTTCTGAAC"
reference = "GTGGGATCTCGCGAAATTCTTTGCCGCACTGGCCCGCGCCAATATCAACATTGTCGCCATTGCTCAGGGATCTTCTGAAC"
print len(read), len(reference)
hmm_test = hmm.HmmModel()
print "Backward:\n\n",
b = count_backward(read, reference, hmm_test)[1]

print '\n\n Forward:\n'
f = count_forward(read, reference, hmm_test)[1]


print "\n Backward:\n", b, '\n Forward:\n', f, '\n'
print "\n Backward:\n", numpy.exp(b), '\n Forward:\n', numpy.exp(f), '\n'
print f/b

