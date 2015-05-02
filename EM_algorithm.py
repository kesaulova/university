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


def write_to_file_matrix(cur_file, matrix):
    """
    Write given matrix in given file
    :param cur_file: already opened file
    :param matrix: two-dimensional matrix
    :return: nothing
    """
    dim_1 = numpy.asarray(matrix).shape[0]
    dim_2 = numpy.asarray(matrix).shape[1]
    for row in xrange(dim_1):
        cur_file.writelines(['\t'.join([str(round(matrix[row, col], 4)) for col in xrange(dim_2)]), '\n'])
    return 0


def write_to_file_array(cur_file, array):
    """
    Write given matrix in given file
    :param cur_file: already opened file
    :param array: two-dimensional matrix
    :return: nothing
    """
    dim_1 = len(array)
    cur_file.writelines(['\t'.join([str(round(array[col], 4)) for col in xrange(dim_1)]), '\n'])
    return 0


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


def get_exp(item):
    """
    Take a log from every element in array
    :param item: array (one or more dimensional)
    :return: ndarray with same shape, with log-elements
    """
    nd_item = numpy.array(item).copy()
    for x in numpy.nditer(nd_item, op_flags=['readwrite']):
        x[...] = numpy.exp(x)
    return nd_item


def hmm_block(reference):
    """
    Process reference string, detect hmm_block for each position of reference
    (reference block - number of HP from begining)
    :param reference: nucleotide reference
    :return: list of length len(reference), where number of each position determine what hmm_block correspond by this
    nucleotide in reference
    """
    result = [0]*len(reference)
    base = reference[0]
    for i in xrange(1, len(reference)):
        if reference[i] == base:
            result[i] = result[i - 1]
        else:
            base = reference[i]
            result[i] = result[i - 1] + 1
    return result


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
            result = log_sum(result, iter_plog([forward_position[i - k, j - 1, prev_state], transition, emission]))
            # print states[prev_state], read_pos, ref_pos, prev_state, k, l, forward[read_pos][ref_pos][prev_state][k][l]
        forward[i, j, k, reference[j].length, 0] = result
        # print "read: ", i, "len: ", k, "ref: ", j, "len: ", l, states[0], forward[i, j, k, l, 0]
        # print i, j, k, l, states[0], forward[i, j, k, l, 0]
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
        # print "read: ", i, "len: ", k, "ref: ", j, "len: ", l, states[2], forward[i, j, k, l, 2]
        # print i, j, k, l, states[2], forward[i, j, k, l, 2]
        return 0

    def process_deletion(i, j, k):
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
        forward[i, j, k, reference[j].length, 1] = result
        return 0

    # start from 0, because there can be insertions and deletions at the begining
    for i in xrange(len(read)):

        for j in xrange(len(reference)):

            if i == j == 0:     # Can't align two empty hp
                continue

            for k in xrange(max_hp_read[i] + 1):

                for l in xrange(2):

                    if l == 0 and k == 0:
                        continue
                    elif l == 0:
                        process_insertion(i, j, k, homopolymer(read[i], k))
                    elif k == 0:
                        process_deletion(i, j, k)
                    elif read[i] == reference[j].base:
                        process_match(i, j, k, reference[j], homopolymer(read[i], k))

            # fill F(i,j,pi)
            for state in states:
                forward_position[i][j][state] = by_iter_slog(numpy.nditer(forward[i, j, :, :, state]))

    check = by_iter_slog(numpy.nditer(forward_position[len(read_tmp), len(reference) - 1, :]))
    # print forward_position[len(read_tmp), len(reference) - 1, :]

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
    # B(n, m, k, 0, Insertion)
    emiss = [model.HMM[len_ref].emission(homopolymer(), homopolymer(read[len_read], t), 'Insertion')
             for t in xrange(1, length_last_hp(read) + 1)]
    backward[len_read, len_ref, 1: length_last_hp(read) + 1, 0, state_index['Insertion']] = emiss
    # B(n, m, k, 1, Match)
    emiss = [model.HMM[len_ref].emission(reference[len_ref], homopolymer(read[len_read], t), 'Match')
             for t in xrange(1, length_last_hp(read) + 1)]
    backward[len_read, len_ref, 1: length_last_hp(read) + 1, reference[len_ref].length, state_index['Match']] = emiss

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

            if i != 0 or j != 0:
                # Count B(i, j, k, l, Match)
                if read[i] == reference[j].base:
                    trans_prob = [model.HMM[j].transition('Match', states[k]) for k in xrange(len(states))]
                    if i != len_read and read[i] == read[i + 1]:
                        trans_prob[2] = float("-inf")
                    value = iter_slog(trans_prob + part_two)
                    for k in xrange(1, max_hp_read_e[i] + 1):
                        backward[i, j, k, reference[j].length, state_index['Match']] = value

                # Count B(i, j, k, l, Deletion)
                trans_prob = [model.HMM[j].transition('Deletion', states[k]) for k in xrange(len(states))]
                value = iter_slog(trans_prob + part_two)
                backward[i, j, 0,  reference[j].length, state_index['Deletion']] = value

                # Count B(i, j, k, l, Insertion)
                trans_prob = [model.HMM[j].transition('Insertion', states[k]) for k in xrange(len(states))]
                if i != len_read and read[i] == read[i + 1]:
                    trans_prob[0] = float("-inf")
                value = iter_slog(trans_prob + part_two)
                for k in xrange(1, max_hp_read_e[i] + 1):
                    backward[i, j, k, 0, state_index['Insertion']] = value

            else:   # i == 0 and j == 0:     # Begin case!
                trans_prob = [model.HMM[0].transition('Begin', states[k]) for k in xrange(len(states))]
                backward[0, 0, 0, 0, 3] = iter_slog(trans_prob + part_two)
                # print trans_prob + part_two
    return backward, backward[0, 0, 0, 0, 3]


def count_missing_variables(model, read_tmp, reference_tmp): #, transition_matrix):
    """
    Count gamma and xi for given read and reference. Variables counts with usage of forward-backward algorithm.
    And update information about transition matrix
    :param model: current HMM
    :param read_tmp: given read
    :param reference_tmp: given reference
    :return: gamma and xi
    """
    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion', 3: 'Begin', 4: 'End'}
    read = ' ' + read_tmp
    reference = [homopolymer()] + nucl_to_hp(reference_tmp)

    max_hp_read = [0] + len_max_hp_end(read_tmp)    # information about the lengthiest HP at each position
    max_hp_ref = [0] + len_max_hp_end(reference_tmp)

    forward, check_forward = count_forward(read_tmp, reference_tmp, model)
    backward, check_backward = count_backward(read_tmp, reference_tmp, model)
    #  check_forward and check_backward must be same
    print "Difference: ", abs(check_forward - check_backward)
    print numpy.exp(check_backward)/numpy.exp(check_forward)
    # shape of forward and backward must also be the same
    # print "Shape: ", forward.shape, backward.shape

    # count first missing variable. Amazing :)
    # Shape of gamma: len_read, len_ref, max_hp_read, max_hp_ref, len(states)
    gamma = forward + backward
    gamma -= check_forward

    # count second missing variable. Need to do it faster - by presenting operations as a vector operations
    # xi(i, j, pi, pi', k, l)
    xi = float("-inf")*numpy.ones(shape=[len(read), len(reference), len(states), len(states), max(max_hp_read) + 1,
                                         max(max_hp_ref) + 1], dtype=float)

    def supp(curr_st, prev_st, i, j, k, l, hp_input, hp_output, check):
        transition = model.HMM[j].transition(states[prev_st], states[curr_st])
        emission = model.HMM[j].emission(hp_input, hp_output, states[curr_st])
        backward_curr = backward[i, j, k, l, curr_st]
        it = numpy.array([[forward[i - k, j - l, k_tmp, l_tmp, prev_state]
               for k_tmp in xrange(max_hp_read[i - k] + 1)] for l_tmp in xrange(2)])
        it = it.flatten()
        it = iter_slog(it)
        xi[i, j, curr_st, prev_st, k, l] = iter_plog([transition, backward_curr, emission,  it, (-1)*check])
        return 0

    for i in xrange(1, len(read)):  # read position, start from 1 (count position from 1)
        for j in xrange(1, len(reference)):     # reference position
            for k in xrange(max_hp_read[i] + 1):     # maximum length of HP, ending at i
                for prev_state in states:   # previous state

                    if k == 0:
                        # Deletion case l = 1, k = 0
                        supp(1, prev_state, i, j, 0, 1, reference[j], homopolymer(), check_forward)
                        continue
                    # Match case: l = 1, k >= 1
                    supp(0, prev_state, i, j, k, 1, reference[j], homopolymer(read[i], k), check_forward)

                    # Insertion case: l = 0, k >= 1
                    supp(2, prev_state, i, j, k, 0, homopolymer(), homopolymer(read[i], k), check_forward)

                    # UPDATE TRANSITION
                    # transition_matrix[prev_state, curr_state] = log_sum(transition_matrix[prev_state, curr_state],
                    # xi[i, j, curr_state, prev_state, k, l])
    return gamma, xi


def update_parameters(training_set, base_model, max_hp_len, b, sigma, hp_freq, len_string):
    """

    :param training_set: set of pairs read-reference
    :param base_model: current HMM model
    :param max_hp_length: model?
    :param b: array with length max_hp_len of scale parameters for Laplace distribution
    :param sigma: scale parameter of log-normal distribution
    :param hp_freq: observed frequencies of homopolymers
    :return:
    """

    def transition_normalize(transition_matrix):
        """
        Counted transition matrix consist of logs of probabilites. Sum of these probabilities must be 1.
        Here get exp() from matrix, normalize it and ln() result.
        :param transition_matrix: matrix from ln of probabilities (non-normalized)
        :return: normalized matrix
        """
        tmp_matrix = numpy.exp(transition_matrix)
        for i in xrange(len(states)):
            if sum(tmp_matrix[i, ]) != 0:   # ATTENTION! Do it because don't count begin/end
                tmp_matrix[i, ] = tmp_matrix[i, ] / sum(tmp_matrix[i, ])
        for x in numpy.nditer(tmp_matrix, op_flags=['readwrite']):
            x[...] = eln(x)
        return tmp_matrix

    def base_call_normalize(base_call):
        """
        Normalize counted array of base call for insertion
        :param base_call: one-dimensional array of log-probabilities.
        :return: normalized array
        """
        tmp_base = numpy.exp(base_call)
        tmp_base = tmp_base / sum(tmp_base)
        for x in numpy.nditer(tmp_base, op_flags=['readwrite']):
            x[...] = eln(x)
        return tmp_base

    def write_params():
        file_tmp = open("Counted_params.txt", 'w')

        # write transition matrix
        file_tmp.writelines("\n Transition matrix: \n")
        write_to_file_matrix(file_tmp, get_exp(transition_matrix))
        for i in xrange(len(transition_matrix[:, 0])):
            file_tmp.writelines(["\n" + str(sum(get_exp(transition_matrix[i, :]))) + "\n"])

        # write insertion base call
        file_tmp.writelines("\n Insertion base call: \n")
        write_to_file_array(file_tmp, get_exp(ins_base_call))
        file_tmp.writelines(["\n" + str(sum(get_exp(ins_base_call))) + "\n"])

        # write b
        file_tmp.writelines("\n Previous b: \n")
        write_to_file_array(file_tmp, b)
        file_tmp.writelines("\n Current b: \n")
        write_to_file_array(file_tmp, new_b)

        # write sigma
        file_tmp.writelines("\n Previous sigma: \n" + str(sigma) + '\n')
        file_tmp.writelines("\n Current sigma: \n" + str(new_sigma) + '\n')

        # write insertion length call
        file_tmp.writelines("\n Length call (insertion): \n")
        write_to_file_array(file_tmp, get_exp(length_call_ins))
        file_tmp.writelines(["\n" + str(sum(get_exp(length_call_ins))) + "\n"])

        # write match length call
        file_tmp.writelines("\n Length call (match): \n")
        write_to_file_matrix(file_tmp, get_exp(length_call_match))
        for i in xrange(len(length_call_match[:, 0])):
            file_tmp.writelines(["\n" + str(sum(get_exp(length_call_match[i, :]))) + "\n"])
        file_tmp.close()
        return 0

    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion', 3: 'Begin', 4: 'End'}
    bases = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    transition_matrix = float("-inf")*numpy.ones(shape=[len(states), len(states)], dtype=float)
    length_call_matrix = float("-inf")*numpy.ones(shape=[max_hp_len + 1, max_hp_len + 1], dtype=float)
    ins_base_call = float("-inf")*numpy.ones(shape=[4], dtype=float)
    counter = 0
    max_hp = 0
    for pair in training_set:   # process every pair of read, reference

        read = pair[0][:len_string]
        reference = pair[1][:len_string]
        # FOR TESTING! Not interesting in alignment of equal string
        # if read == reference:
        #     continue
        print "\n Step:", counter, '\n'
        if counter == 45:
            break
        counter += 1
        print "Read:     ", read, "\nReference:", reference

        # xi shape: [read_pos, ref_pos, prev_state, curr_state, hp_read_len, hp_ref_len]
        # gamma shape: [read_pos, ref_pos, hp_read_len, hp_ref_len, state]
        gamma, xi = count_missing_variables(base_model, read, reference)

        max_hp_read = max(len_max_hp_end(read))    # information about the lengthiest HP
        max_hp_ref = max(len_max_hp_end(reference))
        if max_hp < max_hp_ref:
            max_hp = max_hp_ref
        # update T (transition matrix), T(pi, pi')
        for previous in states:
            for current in states:
                transition_matrix[previous, current] = by_iter_slog(numpy.nditer(xi[:, :, previous, current, :, :]))

        # update L(k,l) - occurrences of length calling
        for k in xrange(max_hp_read + 1):     # maximum length of HP. All of not-useful will be -inf
            for l in xrange(max_hp_ref + 1):
                # print l, k, "Shape: ", length_call_matrix.shape
                length_call_matrix[l, k] = by_iter_slog(numpy.nditer(gamma[:, :, k, l, :]))

        # update length call for insertion. gamma[i + 1...], because here count read from 0, in gamma from 1
        for i in xrange(len(read)):
            ins_base_call[bases[read[i]]] = log_sum(ins_base_call[bases[read[i]]],
                                                    by_iter_slog(numpy.nditer(gamma[i + 1, :, 1:, 0, 2])))
        gamma = xi = None


    transition_matrix = transition_normalize(transition_matrix)
    ins_base_call = base_call_normalize(ins_base_call)
    length_call_match, length_call_ins, new_b, new_sigma = update_length_call_parameters(length_call_matrix,
                                                                                       b, max_hp_len, hp_freq, sigma)
    write_params()
    return ins_base_call, length_call_match, length_call_ins, transition_matrix


def update_length_call_parameters(length_call_matrix_ln, b, max_length_hp, p_k, sigma):
    """
    :param length_call_matrix: matrix with log-probabilities
    :param b: one-dimensional list with parameters of Laplace distrib
    :param max_length_hp: maximum length of HP
    :param p_k: homopolymers length array (observed from genome)
    :param sigma: parameter of log-normal distribution
    :return: udpated length call matrix, length cal insertion array
    """
    # length_call_matrix = numpy.exp(length_call_matrix_ln)
    length_call_matrix = get_exp(length_call_matrix_ln)
    f_start = 0.0001
    f_end = max_length_hp + 1
    f = numpy.arange(f_start, f_end, 0.1)

    def count_p_f_l(b_scale):
        """
        Count p(f|l) - Laplace distribution, probability of flow intensity f, when input HP have length l
        :param b: parameter of scale
        :return: 0
        """
        tmp = numpy.zeros(shape=[len(f), max_length_hp + 1], dtype=float)
        for ff in xrange(len(f)):
            for l in xrange(1, max_length_hp + 1):   # l = 0 count by log-normal distribution
                tmp[ff, l] = laplace.pdf(f[ff], loc=l, scale=b_scale[l])
        return tmp

    def count_z_f():
        """
        for each f count Z = \sum_{k}p(f | k) * p(k)
        """
        tmp = numpy.zeros(shape=[len(f)], dtype=float)     # Coefficient of normalize Z = \sum_{k}p(f | k) * p(k)
        for i in xrange(len(f)):
            tmp[i] = sum(p_f_l[i, :]*p_k[:len(p_f_l[i, :])])
        return tmp

    def count_p_k_f():
        """
        Count p(k|f) - probability of observing HP length k from flow intensity f
        p(k|f) = p(f|k)*p(k)/Z
        """
        tmp = numpy.zeros(shape=[max_length_hp + 1, len(f)], dtype=float)     # p(k | f)
        it = numpy.nditer(tmp, flags=['multi_index'], op_flags=['writeonly'])
        while not it.finished:
            k_tmp = it.multi_index[0]
            f_tmp = it.multi_index[1]
            it[0] = p_f_l[f_tmp, k_tmp] * p_k[k_tmp] / z_f[f_tmp]
            it.iternext()
        # tmp = numpy.zeros(shape=[max_length_hp + 1, len(f)], dtype=float)     # p(k | f)
        # for k in xrange(1, max_length_hp + 1):
        #     for ff in xrange(len(f)):
        #         tmp[k, ff] = p_f_l[ff, k] * p_k[k] / z_f[ff]
        return tmp

    def count_p_f_zero(sigma):
        tmp = []
        for ff in xrange(len(f)):
            tmp.append(lognorm.pdf(f[ff], 1, loc=0, scale=sigma))
        return tmp

    def expectation_step():
        """
        Count p(f|k,l) (p_f_k_l and p_f_k_zero) (See supplementary)
        """
        tmp = numpy.zeros(shape=[len(f), max_length_hp + 1, max_length_hp + 1], dtype=float)     # p(f | k, l)
        tmp_zero = numpy.zeros(shape=[len(f), max_length_hp + 1], dtype=float)     # p(f | k, 0)
        for k in xrange(1, max_length_hp + 1):
            tst = [p_k_f[k, i] * p_f_zero[i] for i in xrange(len(f))]
            if sum(tst) != 0:
                tst = tst / sum(tst)
            tmp_zero[:, k] = tst[:]
            for l in xrange(1, max_length_hp + 1):
                tst = [p_k_f[k, i]*p_f_l[i, l] for i in xrange(len(f))]
                if sum(tst) != 0:
                    tst = tst / sum(tst)
                tmp[:, k, l] = tst[:]  # p(k|f) * p(f|l)
        return tmp_zero, tmp

    def prod(x, y):
            assert (len(x) == len(y)), "773"
            res = 0
            for i in xrange(len(x)):
                res += x[i] * y[i]
            return res

    def count_b():
        """
        Count parameters b of Laplace distribution. l > 0.
        Write result in created before new_b
        :return: 0
        """

        tmp_b = [0] * (max_length_hp + 1)
        # count b for l > 0
        for l in xrange(1, max_length_hp + 1):
            numerator = 0
            denominator = 0
            for k in xrange(1, max_length_hp + 1):
                tmp_num = [p_f_k_l[ff, k, l] * abs(f[ff] - l) for ff in xrange(len(f))]
                numerator = numerator + length_call_matrix[l, k] * sum(tmp_num)
                denominator = denominator + sum(p_f_k_l[:, k, l]) * length_call_matrix[l, k]
            # numerator_tmp = 0
            # denominator_tmp = 0
            # for ff in xrange(len(f)):
            #     for k in xrange(1, max_length_hp + 1):
            #         temp = p_f_k_l[ff, k, l]*length_call_matrix[l, k]
            #         numerator_tmp += temp * abs(f[ff] - l)
            #         denominator_tmp += temp
            print round(numerator, 4), round(denominator, 4)
            tmp_b[l] = numerator / denominator
        return tmp_b

    def count_sigma():
        """
        Count parameter of log-normal distribution.
        :return: counted parameter
        """
        numerator_tmp = sum([prod(p_f_k_zero[ff, :], length_call_matrix[0, :])*eln(f[ff])**2 for ff in xrange(len(f))])
        denominator_tmp = sum([prod(p_f_k_zero[ff, :], length_call_matrix[0, :]) for ff in xrange(len(f))])
        numerator = 0
        denominator = 0
        for ff in xrange(len(f)):
            for k in xrange(max_length_hp):
                element = p_f_k_zero[ff, k] * length_call_matrix[0, k]
                denominator += element
                numerator += (element * eln(f[ff])**2)
        print "Count sigma ", round(numerator - numerator_tmp, 4), round(denominator - denominator_tmp, 4)
        return math.sqrt(numerator / denominator)

    def count_length_call_match(max_hp_length, scale, pk):
        """
        Count length-call matrix, based on updated scale parameter of laplace distribution
        :param max_hp_length: maximum length of hp
        :param scale: updated scale parameter of Laplace distribution
        :return: matrix
        """
        result = numpy.zeros(shape=[max_hp_length, max_hp_length], dtype=float)

        def lcall(x, l, k):
            """
            Integrated function
            :param x: f, flow intensity
            :param l: length of input hp
            :param k: length of output hp
            :return: counted function
            """
            num = laplace.pdf(x, loc=l, scale=scale[l]) * laplace.pdf(x, loc=k, scale=scale[k]) * pk[k]
            denom = sum([pk[i] * laplace.pdf(x, loc=i, scale=scale[i]) for i in xrange(1, max_hp_length + 1)])
            return num/denom

        def normalize(item, max_len):
            """
            Normalize length call matrix (sum values in one row must be 1)
            :param item: square matrix
            :param max_len: number of rows/columns
            :return:
            """
            for i in xrange(max_len):
                item[i, ] = item[i, ] / sum(item[i, ])
            return item

        for l in xrange(1, max_hp_length + 1):
            for k in xrange(1, max_hp_length + 1):
                result[l - 1, k - 1] = quad(lcall, 0, max_hp_length, args=(l, k))[0]
        result = normalize(result, max_hp_length)
        return result

    def count_length_insertion(max_hp_length, sigma_scale, b_scale, pk):
        """
        Count length call in case zero-length of input hp
        :param max_hp_length: maximum hp length
        :param sigma_scale: scale parameter for log-normal distribution
        :param b_scale: scale parameters (an array) for Laplace distribution
        :return: an array with probabilities of calling certain length from 0
        """
        result = numpy.zeros(shape=[max_hp_length], dtype=float)

        def lcall(x, k):
            num = lognorm.pdf(x, 1, loc=0, scale=sigma_scale) * laplace.pdf(x, loc=k, scale=b_scale[k]) * pk[k]
            denom = sum([pk[i] * laplace.pdf(x, loc=i, scale=b_scale[i]) for i in xrange(1, max_hp_length + 1)])
            return num/denom

        for k in xrange(1, max_hp_length + 1):
            result[k - 1] = quad(lcall, 0, max_hp_length, args=(k))[0]
        result = result / sum(result)
        return result

    def write_evr():
        """
        Write to files all counted params
        :return:
        """
        file_tmp = open("p_f_k_l.txt", 'w')
        for ff in xrange(0, len(f), len(f)/max_length_hp):
            file_tmp.writelines("\n p(f | k, l), f = " + str(f[ff]) + '\n')
            write_to_file_matrix(file_tmp, p_f_k_l[ff, :, :])

        file_tmp.close()

        file_tmp = open("update_parameters.txt", 'w')
        for i in xrange(0, len(f), len(f)/max_length_hp):
            file_tmp.writelines("\n p(f | k, l), f = " + str(f[i]) + '\n')
            write_to_file_matrix(file_tmp, p_f_k_l[i, :, :])

        file_tmp.writelines('\n')

        file_tmp.writelines("\n p(f | k, 0)\n")
        for i in xrange(0, len(f), len(f)/max_length_hp):
            write_to_file_array(file_tmp, p_f_k_zero[i, :])

        file_tmp.writelines('\n')
        file_tmp.writelines("\n p(f | l)\n")
        for i in xrange(0, len(f), len(f)/max_length_hp):
            write_to_file_array(file_tmp, p_f_l[i, :])

        file_tmp.writelines('\n')
        file_tmp.writelines("Z_f\n")
        write_to_file_array(file_tmp, z_f)

        file_tmp.writelines('\n')
        file_tmp.writelines("\n p(k | f)\n")
        write_to_file_matrix(file_tmp, numpy.asarray([p_k_f[:, j] for j in xrange(0, len(f), len(f)/max_length_hp)]))

        file_tmp.writelines('\n')
        file_tmp.writelines("\n p(f | 0)\n")
        write_to_file_array(file_tmp, numpy.asarray([p_f_zero[i] for i in xrange(0, len(f), len(f)/max_length_hp)]))
        file_tmp.close()

    p_f_l = count_p_f_l(b)     # p(f | l)
    z_f = count_z_f()      # Coefficient of normalize Z = \sum_{k}p(f | k) * p(k)
    p_k_f = count_p_k_f()       # p(k | f)
    p_f_zero = count_p_f_zero(sigma)    # p(f | 0)
    p_f_k_zero, p_f_k_l = expectation_step()     # p(f | k, 0),  p(f | k, l)
    write_evr()

    new_b = count_b()

    for i in xrange(len(new_b)):
        if math.isnan(new_b[i]):
            new_b[i] = new_b[i - 1] + (new_b[i - 1] - new_b[i - 2])/2.0

    new_sigma = count_sigma()

    length_call_match = count_length_call_match(max_length_hp, new_b, p_k)
    length_call_insertion = count_length_insertion(max_length_hp, new_sigma, new_b, p_k)
    return get_eln(length_call_match), get_eln(length_call_insertion), new_b, new_sigma
