import hmm
import numpy
import addmath
import math

eln = addmath.eln
exp = addmath.eexp
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
    for i in range(len(read) - 2, -1, -1):
        if read[i] == base:
            result += 1
        else:
            break
    return result


def len_max_hp_end(sequence):
    """
    Create list were at each positions is length of lengthest hmm.homopolymer, ended at these position
    :param sequence: nucleotide sequence
    :return: list with numbers, indicated length of the longest hmm.homopolymer, ended at these position
    """
    result = [1]*len(sequence)
    base = sequence[0]
    for i in range(1, len(sequence)):
        if sequence[i] == base:
            result[i] = result[i - 1] + 1
        else:
            base = sequence[i]
            result[i] = 1
    return result


def len_max_hp_start(sequence):
    """
    Create list were at each positions is length of lengthest hmm.homopolymer, starting at these position
    :param sequence: nucleotide sequence
    :return: list with numbers, indicated length of the longest hmm.homopolymer, ended at these position
    """
    result = [1]*len(sequence)
    base = sequence[len(sequence) - 1]
    for i in range(len(sequence) - 2, -1, -1):
        if sequence[i] == base:
            result[i] = result[i + 1] + 1
        else:
            base = sequence[i]
            result[i] = 1
    return result


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
    for i in range(1, len(reference)):
        if reference[i] == base:
            result[i] = result[i - 1]
        else:
            base = reference[i]
            result[i] = result[i - 1] + 1
    return result


# I count F and B from 1! Read and reference. In HMM 0 block carry information about initial probability distribution
def count_forward(read_tmp, reference_tmp, model):
    """
    Count forward variable F(i,j,k,l,pi) - is the probability summarizing over all possible alignments ending at
    the hidden state pi between the prefixes read[ 1:i] and reference[1: j] in which a k bp homopolymer at the
    read position i is aligned to a l bp homopolymer in the reference position j .
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

    max_hp_read = [0] + len_max_hp_end(read) + [0]
    len_hp_ref = len_max_hp_end(reference_tmp)   # information about the length of HP in reference
    forward = float("-inf")*numpy.ones(shape=[len(read), len(reference), len(states), max(max_hp_read) + 1,
                                              max(len_hp_ref) + 1], dtype=float)
    print forward.shape
    forward_position = float("-inf")*numpy.ones(shape=[len(read), len(reference), 5], dtype=float)
    print forward_position.shape
    forward_position[0][0][3] = 0

    def process_match(i, j, k, l, hp_input, hp_output):
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
            transition = eln(model.HMM[j - 1].transition(states[prev_state], 'Match'))
            emission = eln(model.HMM[j].emission(hp_input, hp_output, 'Match'))
            result = log_sum(result, iter_plog([forward_position[i - k][j - 1][prev_state], transition, emission]))
            # print states[prev_state], read_pos, ref_pos, prev_state, k, l, forward[read_pos][ref_pos][prev_state][k][l]
        forward[i][j][0][k][l] = result
        # print "read: ", i, "len: ", k, "ref: ", j, "len: ", l, states[0], forward[i][j][0][k][l]
        # print i, j, k, l, states[0], forward[i][j][0][k][l]
        return 0

    def process_insertion(i, j, k, l, hp_output):
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
            transition = eln(model.HMM[j].transition(states[prev_state], 'Insertion'))
            emission = eln(model.HMM[j].emission(homopolymer(), hp_output, 'Insertion'))
            result = log_sum(result, iter_plog([forward_position[i - k, j, prev_state], transition, emission]))
        forward[i, j, 2, k, l] = result
        # print "read: ", i, "len: ", k, "ref: ", j, "len: ", l, states[2], forward[i][j][2][k][l]
        # print i, j, k, l, states[2], forward[i][j][2][k][l]
        return 0

    def process_deletion(i, j, k, l, hp_input):
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
            transition = eln(model.HMM[j - 1].transition(states[prev_state], 'Deletion'))
            emission = eln(model.HMM[j].emission(hp_input, homopolymer(), 'Deletion'))
            result = log_sum(result, iter_plog([forward_position[i][j - 1][prev_state], transition, emission]))
            #print states[prev_state], read_pos, ref_pos, prev_state, k, l, forward_position[read_pos][ref_pos][prev_state]
        forward[i, j, 1, k, l] = result
        # print "read: ", i, "len: ", k, "ref: ", j, "len: ", l, states[1], forward[i][j][1][k][l]
        # print i, j, k, l, states[1], forward[i][j][1][k][l]
        return 0

    # start from 0, because there can be insertions and deletions at the begining
    for i in range(len(read)):

        for j in range(len(reference)):
            if i == j == 0:
                continue

            for k in range(max_hp_read[i] + 1):

                for l in range(2):
                    #print "169, ", i, j, k, l
                    if l == 0 and k == 0:
                        continue
                    elif l == 0:
                        process_insertion(i, j, k, l, homopolymer(read[i], k))
                    elif k == 0:
                        process_deletion(i, j, k, l, reference[j])
                    elif read[i] == reference[j].base:
                        process_match(i, j, k, l, reference[j], homopolymer(read[i], k))

            # fill F(i,j,pi)
            for state in states:
                forward_position[i][j][state] = by_iter_slog(numpy.nditer(forward[i, j, state, :, :]))
                #if not math.isnan(forward_position[i][j][state]):
                #    print "---------------------------", i, j, states[state], forward_position[i][j][state]
    print len(read_tmp), len(reference_tmp)
    check = by_iter_slog(numpy.nditer(forward_position[len(read_tmp), len(reference) - 1, :]))
    print check
    return forward


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
    reference = [homopolymer()]
    reference.extend(nucl_to_hp(reference_tmp))
    len_ref = len(reference) - 1  # true length
    len_read = len(read_tmp)
    print "Len_read: ", len_read, "len_reference: ", len_ref
    # position and create len_sequence to remember true length
    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion', 3: 'Begin', 4: 'End'}
    state_index = {'Match': 0, 'Deletion': 1, 'Insertion': 2, 'Begin': 3, 'End': 4}
    max_hp_read_s = [0] + len_max_hp_start(read_tmp)  # because read start from ' '

    # information about the lengthest HP, ending from this position
    max_hp_read_e = [0] + len_max_hp_end(read_tmp)  # because read start from ' '
    max_hp_ref_e = [0] + len_max_hp_end(reference_tmp)  # because reference start from ' '

    backward = float("-inf")*numpy.ones(shape=[len_read + 1, len_ref + 1, max(max_hp_read_e) + 1,
                                               max(max_hp_ref_e) + 1,  5], dtype=float)
    print "Size: ", backward.nbytes
    print "Shape: ", backward.shape

    # initialize
    # B(i, m, 0, 1, Deletion)
    backward[len_read, len_ref, 0, reference[len_ref].length, state_index['Deletion']] = 0
    # B(n, m, k, 0, Insertion)
    emiss = [eln(model.HMM[len_ref].emission(homopolymer(), homopolymer(read[len_read], t), 'Insertion'))
             for t in range(1, length_last_hp(read) + 1)]
    backward[len_read, len_ref, 1: length_last_hp(read) + 1, 0, state_index['Insertion']] = emiss
    print 'Insertion', len_read, len_ref, length_last_hp(read), emiss
    # B(n, m, k, 1, Match)
    emiss = [eln(model.HMM[len_ref].emission(reference[len_ref], homopolymer(read[len_read], t), 'Match'))
             for t in range(1, length_last_hp(read) + 1)]
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
        emiss = [eln(model.HMM[j + 1].emission(hp_input, homopolymer(read_base, t), 'Match')) for t in range(1, hp_len + 1)]
        bck = [backward[i + t, j + 1, t, reference[j + 1].length, st_index] for t in range(1, hp_len + 1)]
        # print 'Match', i + hp_len, j + 1, hp_len, reference[j + 1].length, backward[i + hp_len, j + 1, hp_len, reference[j + 1].length, st_index]
        result = [log_product(x, y) for x, y in zip(emiss, bck)]
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
        # print 'Deletion',i, j + 1, 0, reference[j + 1].length, backward[i, j + 1, 0, reference[j + 1].length, state_index['Deletion']]
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
        emiss = [eln(model.HMM[j].emission(hp_input, homopolymer(read_base, t), 'Insertion')) for t in range(1, hp_len + 1)]
        bck = [backward[i + t, j, t, 0, st_index] for t in range(1, hp_len + 1)]
        # print 'Insertion', i + hp_len, j, hp_len, reference[j].length, backward[i + hp_len, j, hp_len, reference[j].length, st_index]
        result = [log_product(x, y) for x, y in zip(emiss, bck)]
        if len(result) == 1:
            return result[0]
        else:
            return iter_slog(result)


    for i in range(len_read, -1, -1):  # read position
    # for i in range(len_read, len_read - 5, -1):  # read position

        for j in range(len_ref, -1, -1):     # reference position
        # for j in range(len_ref, len_ref - 5, -1):     # reference position
            if j == len_ref and i == len_read:
                continue

            # First count \sum\limits{i,j} p(beta_(i + di)|alpha_(j + dj), pi')*p(k|l, pi')*B(i + di, j + dj, k, pi').
            # It will be vector of length 3. Then, for each state, we create vector of transition probs,
            # element-wise multiply them and then get sum.
            part_two = [(-1)*numpy.inf] * len(states)

            if j != len_ref and i != len_read and read[i + 1] == reference[j + 1].base:
                part_two[0] = process_match(i, j)
            if j != len_ref:
                part_two[1] = process_deletion(i, j)
            if i != len_read:
                part_two[2] = process_insertion(i, j)

            # Count B(i, j, k, l, Match)
            if read[i] != reference[j].base:
                value = (-1)*numpy.inf
            else:
                trans_prob = [eln(model.HMM[j].transition('Match', states[k])) for k in range(len(states))]
                if i != len_read and read[i] == read[i + 1]:
                    trans_prob[2] = float("-inf")
                value_1 = [log_product(x, y) for x, y in zip(trans_prob, part_two)]
                value = iter_slog(value_1)


            for k in range(1, max_hp_read_e[i] + 1):
                backward[i, j, k, reference[j].length, state_index['Match']] = value
            if i < 5 and j < 5:
                print "---------------------------", i, j, 'Match', "{0:.2f}".format(value), [round(x, 3) for x in part_two], [round(x, 3) for x in [eln(model.HMM[j].transition('Match', states[k])) for k in range(len(states))]]

            # Count B(i, j, k, l, Deletion)
            trans_prob = [eln(model.HMM[j].transition('Deletion', states[k])) for k in range(len(states))]
            value = [log_product(x, y) for x, y in zip(trans_prob, part_two)]
            value = iter_slog(value)
            if i < 5 and j < 5:
                print "---------------------------", i, j, 'Deletion', "{0:.2f}".format(value), [round(x, 3) for x in part_two], [round(x, 3) for x in trans_prob]
            backward[i, j, 0,  reference[j].length, state_index['Deletion']] = value

            # Count B(i, j, k, l, Insertion)
            trans_prob = [eln(model.HMM[j].transition('Insertion', states[k])) for k in range(len(states))]
            if i != len_read and read[i] == read[i + 1]:
                trans_prob[0] = float("-inf")
            value = [log_product(x, y) for x, y in zip(trans_prob, part_two)]
            value = iter_slog(value)
            if i < 5 and j < 5:
                print "---------------------------", i, j, 'Insertion', "{0:.2f}".format(value), [round(x, 3) for x in part_two], [round(x, 3) for x in trans_prob]
            for k in range(1, max_hp_read_e[i] + 1):
                backward[i, j, k, 0, state_index['Insertion']] = value
    check = by_iter_slog(numpy.nditer(backward[0, 0, :, :, :]))
    print check
    return backward


def count_missing_variables(model, read_tmp, reference_tmp, transition_matrix):
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
    reference = ' ' + reference_tmp

    max_hp_read = [0]
    max_hp_read.extend(len_max_hp_end(read_tmp))    # information about the lengthest HP at each position
    max_hp_read.append(0)
    max_hp_ref = [0]
    max_hp_ref.extend(len_max_hp_end(reference_tmp))
    max_hp_ref.append(0)
    ref_hmm_blocks = hmm_block(reference)     # information about HMM blocks

    forward, check_forward = count_forward(read_tmp, reference_tmp, model)
    backward, check_backward = count_backward(read_tmp, reference_tmp, model)
    #  check_forward and check_backward must be same
    print "Difference: ", abs(check_forward - check_backward)
    # shape of forward and backward must also be the same
    print "Shape: ", forward.shape, backward.shape

    # count first missing variable. Amazing :)
    gamma = forward + backward
    gamma -= check_forward
    print "Shape of gamma: ", gamma.shape

    # count second missing variable. Need to do it faster - by presenting operations as a vector operations
    xi = float("-inf")*numpy.ones(shape=[len(read), len(reference), 5, 5, max(len_max_hp_start(read)) + 1,
                           max(len_max_hp_start(reference)) + 1],
                                        dtype=float)
    for i in range(len(read)):  # read position
        for j in range(len(reference)):     # reference position
            for curr_state in states:   # current state
                for k in range(max_hp_read[i] + 1):     # maximum length of HP, ending at i

                    # impossible cases
                    if (curr_state == 0 and k == 0) or (curr_state == 1 and k != 0) or (curr_state == 2 and k == 0):
                        continue

                    for l in range(max_hp_ref[j] + 1):     # maximum length of HP, ending at j

                        if (curr_state == 0 and l == 0) or (curr_state == 1 and l == 0) or (curr_state == 2 and l != 0):
                            continue

                        for prev_state in states:   # previous state
                            transition = eln(model.HMM[ref_hmm_blocks[j]].transition(states[prev_state],
                                                                                     states[curr_state]))
                            backward_curr = backward[i, j, k, l, curr_state]
                            it = [log_product(forward[i - k, j - l, k_tmp, l_tmp, prev_state],
                                         eln(model.HMM[ref_hmm_blocks[j - l]].emission(homopolymer(read[i - k], k_tmp),
                                        homopolymer(reference[j - l], l_tmp), states[prev_state])))
                                  for k_tmp in range(max_hp_read[i - k] + 1) for l_tmp in range(max_hp_ref[j - l] + 1)]
                            it = iter_slog(it)

                            xi[i, j, curr_state, prev_state, k, l] = iter_plog([transition, backward_curr, it])

                            # UPDATE TRANSITION
                            transition_matrix[prev_state, curr_state] = log_sum(transition_matrix[prev_state, curr_state], xi[i, j, curr_state, prev_state, k, l])
    xi -= check_forward
    return gamma, xi


def update_parameters(training_set, base_model, max_hp_length):
    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion', 3: 'Begin', 4: 'End'}
    transition_matrix = float("-inf")*numpy.ones(shape=[len(states), len(states)], dtype=float)
    length_call = float("-inf")*numpy.ones(shape=[max_hp_length + 1, max_hp_length + 1], dtype=float)
    # base_call =

    for pair in training_set:   # process every pair of read, reference
        read = pair[0]
        reference = pair[1]
        # function get transition_matrix by reference, yes?
        gamma, xi = count_missing_variables(base_model, read, reference, transition_matrix)

        # information about length of nucleotides
        max_hp_read = [0]
        max_hp_read.extend(len_max_hp_end(read))    # information about the lengthest HP at each position
        max_hp_ref = [0]
        max_hp_ref.extend(len_max_hp_end(reference))

        # update matrix parameters
        # first - update T
        for i in range(len(read)):  # read position
            for j in range(len(reference)):     # reference position
                for k in range(max_hp_read[i]):     # maximum length of HP. All of unuseful will be -inf
                    for l in range(max_hp_ref[j]):
                        length_call[k, l] = iter_slog([length_call[k, l], gamma[i, j, k, l, :]])
        gamma = xi = None


def update_length_call_parameters(length_call_matrix, b, max_length_hp, p_k, sigma):
    f_start = 0
    f_end = max_length_hp + 1
    f = numpy.arange(f_start, f_end, 0.01)
    p_f_l = numpy.zeros(shape=[len(f), max_length_hp + 1], dtype=float)     # p(f | l)
    p_k_f = numpy.zeros(shape=[max_length_hp + 1, len(f)], dtype=float)     # p(k | f)
    z_f = numpy.zeros(shape=[len(f)], dtype=float)     # Coefficient of normalize Z = \sum_{k}p(f | k) * p(k)
    p_f_k_l = numpy.zeros(shape=[len(f), max_length_hp + 1, max_length_hp + 1], dtype=float)     # p(f | k, l)
    p_f_zero = [0]*len(f)   # p(f | 0)
    new_b = [0] * (max_length_hp + 1)

    def count_p_f_l(b):
        """
        Count p(f|l) - Laplace distribution, probability of flow intensity f, when input HP have length l
        :param b: parameter of scale
        :return: 0
        """
        for ff in range(len(f)):
            for l in range(1, max_length_hp + 1):   # l = 0 count by log-normal distribution
                const = 1 / (2 * b[l])
                p_f_l[ff, l] = const * exp((-2)*abs(ff - l)*const)
        return 0

    def count_z_f():
        """
        for each f count Z = \sum_{k}p(f | k) * p(k)
        """
        for i in xrange(len(f)):
            z_f[i] = sum([p_f_l[i, k]*p_k[k] for k in range(1, max_length_hp + 1)])
        return 0

    def count_p_k_f():
        """
        Count p(k|f) - probability of observing HP length k from flow intensity f
        p(k|f) = p(f|k)*p(k)/Z
        """
        for k in range(1, max_length_hp + 1):
            for ff in range(len(f)):
                p_k_f[k, ff] = p_f_l[ff, k] * p_k[k] / z_f[ff]
        return 0

    def count_p_f_zero():
        const = 1 / (sigma * math.sqrt(2 * math.pi))
        for ff in range(len(f)):
            p_f_zero = exp((-1)*(eln(f[ff])**2*const / sigma)) * const / f[ff]
        return 0

    def expectation_step():
        """
        fill p_f_k_l (See supplementary)
        """
        for k in range(1, max_length_hp + 1):
            for l in range(1, max_length_hp + 1):
                temp = [p_k_f[k, f_tmp]*p_f_l[f_tmp, l] for f_tmp in f]     # p(k|f)*p(f|l)
                normalize = sum(temp)
                temp = [temp[i]/normalize for i in range(len(temp))]
                for ff in range(len(f)):
                    p_f_k_l[f[ff], k, l] = temp[ff]
        return 0

"""
    count_p_f_l(b)
    count_z_f()
    count_p_k_f()
    expectation_step()
    count_p_f_zero()

    # count b for l > 0
    for l in range(1, max_length_hp + 1):
        numerator = 0.0
        denominator = 0.0
        for ff in xrange(f):
            for k in range(1, max_length_hp + 1):
                temp = p_f_k_l[ff, k, l]*length_call_matrix[k, l]
                numerator += temp * abs(ff - l)
                denominator += temp
        new_b[l] = numerator / denominator

    # count sigma

    for k in range(1, max_length_hp + 1):
        numerator = 0.0
        denominator = 0.0
        for ff in xrange(len(f)):
            temp = p_f_zero[ff, k]*length_call_matrix[k, 0]
            numerator += temp * (eln(f[ff]))**2
            denominator += temp
"""


read = "TGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAGAGTGTCTGATAGCAGCTTCTGAACTGGTTACCTGCCGTGAGTAAATTAAAATTTTATTGACTTAGGTCACTAAATACTTTAACCAATATAGGCATAGCGACACAGACAGATAAAATTACAGAGTACACAACATCCATGAAACGACATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGCTTTTTTTTTCGACCAAGGTAACGAGGTAACCAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGC"
reference = "TGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGCTTCTGAACTGGTTACCTGCCGTGAGTAAATTAAAATTTTATTGACTTAGGTCACTAAATACTTTAACCAATATAGGCATAGCGCACAGACAGATAAAAATTACAGAGTACACAACATCCATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGC"


def main():
    """
    Implement training of HMM.
    :param sample_data: {(r^d, t^d), d = 1..N} - N pair of reads and mapped reference segments
    :param model: start model
    :return:
    """
    hmm_test = hmm.HmmModel()
    read = "TGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAGAGTGTCTGATAGCAGCTTCTGAACTGGTTACCTGCCGTGAGTAAATTAAAATTTTATTGACTTAGGTCACTAAATACTTTAACCAATATAGGCATAGCGACACAGACAGATAAAATTACAGAGTACACAACATCCATGAAACGACATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGCTTTTTTTTTCGACCAAGGTAACGAGGTAACCAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGC"
    reference = "TGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGCTTCTGAACTGGTTACCTGCCGTGAGTAAATTAAAATTTTATTGACTTAGGTCACTAAATACTTTAACCAATATAGGCATAGCGCACAGACAGATAAAAATTACAGAGTACACAACATCCATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGC"
    print read[359:]
    count_forward(read, reference, hmm_test)
    count_backward(read, reference, hmm_test)

    return 0

main()