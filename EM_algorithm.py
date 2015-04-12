import hmm
import numpy
import addmath
import math
from scipy.stats import laplace, lognorm
from scipy.integrate import quad
import re

eln = addmath.eln
eexp = addmath.eexp
log_product = addmath.log_product
log_sum = addmath.log_sum
iter_plog = addmath.iter_plog
iter_slog = addmath.iter_slog
homopolymer = hmm.homopolymer
nucl_to_hp = hmm.nucleotide_to_homopolymer
by_iter_slog = addmath.by_iter_slog


def form_dataset():

    def split_cigar(cigar, read, reference, quality_string):
        """
        Use sigar string to delete S and other unuseful symbols and detect reference, corresponding to read.
        :param cigar:
        :param read: read sequence
        :param reference: reference sequence. Form like reference[start_pos: start_pos + read_len + 150]
        :param quality_string: quality string, associated with read
        :return: updated read, reference, quality (as a list with numbers), length of reference
        """
        read_edit = ''
        reference_edit = ''
        quality_list = []
        insertions = []
        alignments = []
        pattern = re.compile('([MIDNSHPX=])')
        values = pattern.split(cigar)[:-1]  # turn cigar into tuple of values
        paired = (values[n:n+2] for n in xrange(0, len(values), 2)) # pair values by twos
        i = 0   # read coordinate index
        g = 0   # reference coordinate index
        for pair in paired:
            l = int(pair[0])    # length of CIGAR event
            t = pair[1]     # type of CIGAR event
            if t == 'M':    # if match, return consecutive coordinates
                alignments.append((g, (i, i + l)))   # (genomic offset, (alignment.start, alignment.end))
                reference_edit += reference[g: g + l]
                read_edit += read[i:i + l]
                quality_list.extend([ord(quality_string[k]) - 33 for k in range(i, i + l)])
                i += l
                g += l
            elif t == 'D':  # skip 'l' number of coordinates in reference
                deletion = '-'*l
                # quality_list.extend([0]*l)
                reference_edit += reference[g:g + l]
                g += l
            elif t == 'I':  # insertion of 'l' length
                insertion = '-'*l
                insertions.append((i, i + l))
                read_edit += read[i:i + l]
                quality_list.extend([ord(quality_string[k]) for k in range(i, i + l)])
                reference_edit += insertion
            elif t == 'N': ## skipped region from the reference
                pass
            elif t == 'S': ## soft clipping (clipped sequences present in SEQ)
                i += l
                pass
            elif t == 'H': ## hard clipping (clipped sequences NOT present in SEQ)
                pass
            elif t == 'P': ## padding (silent deletion from padded reference)
                pass
            elif t == '=': ## sequence match
                pass
            elif t == 'X': # sequence mismatch
                pass
        return read_edit, reference_edit, quality_list, g

    def test_count_b(max_len):
        """
        Take parameters from article and count b. b = c_0 + c_1*l^c_2
        :param max_length: maximum length of homopolymer
        :return: array of length max_length + 1 - for convenience call b[l]
        """
        c_0 = 0.665997
        c_1 = 0.0471694
        c_2 = 1.23072
        res = [0]
        for l in range(max_len):
            res += [c_0 + c_1 * l**c_2]
        return res

    def write_set(read, reference, quality, file):
        """
        Write training set to file
        :param read: read string
        :param reference: reference string
        :param quality: quality array
        :param file: file to write
        :return:
        """
        tmp = read + '\n' + reference + '\n' + quality + '\n\n'
        file.write(tmp)
        return 0

    # temporary!!!
    def count_occurrences(ref_str):
        length_distributions = {'A': [0]*15, 'C': [0]*15, 'G': [0]*15, 'T': [0]*15}
        hp_length = 1
        current_base = ref_str[0]
        for i in xrange(1, len(ref_str)):
            if ref_str[i] == current_base:
                hp_length += 1
            else:
                if hp_length >= 15:
                    print 'Length >= 15!', hp_length
                    continue
                length_distributions[current_base][hp_length] += 1
                current_base = ref_str[i]
                hp_length = 1
        for key in length_distributions:
            length_distributions[key] = [item/float(sum(length_distributions[key])) for item in length_distributions[key]]
        # print "length distrib", length_distributions
        return length_distributions

    # open fasta-file
    fasta_string = '0'
    for line in open("DH10B-K12.fasta", 'r'):
        if line[0] != '>':
            fasta_string += line.rstrip('\n')

    training_set = []
    fasta_length = len(fasta_string)
    observed_freq = count_occurrences(fasta_string[1:])
    b_scale = test_count_b(15)
    tr_set = open("training_set.txt", 'w')
    i = 0   # counter for number of item in training set
    len_train_set = 1000
    for read in open('B22-730.sam'):
        if i > len_train_set:
            break
        else:
            i += 1
        if read[0] == "@":
            continue
        samRecordFields = re.split('\t', read.rstrip('\n'))
        read_seq = samRecordFields[9]
        pos = int(samRecordFields[3])
        if pos + len(read_seq) + 150 < fasta_length:
            read, reference, read_quality, end = split_cigar(samRecordFields[5], read_seq,
                                            fasta_string[pos:pos + len(read_seq) + 150], samRecordFields[10])
            if read_quality != [] and len(read) == len(samRecordFields[10]):
                write_set(read, fasta_string[pos:pos + end], samRecordFields[10], tr_set)
                training_set.append([read, reference, samRecordFields[10]])

        else:
            continue
    tr_set.close()
    return training_set, observed_freq['A'], b_scale


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
    forward = float("-inf")*numpy.ones(shape=[len(read), len(reference), max(max_hp_read) + 1,
                                              max(len_hp_ref) + 1, len(states)], dtype=float)
    # print forward.shape
    forward_position = float("-inf")*numpy.ones(shape=[len(read), len(reference), 5], dtype=float)
    # print forward_position.shape
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
            transition = model.HMM[j - 1].transition(states[prev_state], 'Match')
            emission = model.HMM[j].emission(hp_input, hp_output, 'Match')
            result = log_sum(result, iter_plog([forward_position[i - k, j - 1, prev_state], transition, emission]))
            # print states[prev_state], read_pos, ref_pos, prev_state, k, l, forward[read_pos][ref_pos][prev_state][k][l]
        forward[i, j, k, l, 0] = result
        # print "read: ", i, "len: ", k, "ref: ", j, "len: ", l, states[0], forward[i, j, k, l, 0]
        # print i, j, k, l, states[0], forward[i, j, k, l, 0]
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
            transition = model.HMM[j].transition(states[prev_state], 'Insertion')
            emission = model.HMM[j].emission(homopolymer(), hp_output, 'Insertion')
            result = log_sum(result, iter_plog([forward_position[i - k, j, prev_state], transition, emission]))
        forward[i, j, k, l, 2] = result
        # print "read: ", i, "len: ", k, "ref: ", j, "len: ", l, states[2], forward[i, j, k, l, 2]
        # print i, j, k, l, states[2], forward[i, j, k, l, 2]
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
            transition = model.HMM[j - 1].transition(states[prev_state], 'Deletion')
            emission = model.HMM[j].emission(hp_input, homopolymer(), 'Deletion')
            result = log_sum(result, iter_plog([forward_position[i, j - 1, prev_state], transition, emission]))
            #print states[prev_state], read_pos, ref_pos, prev_state, k, l, forward_position[read_pos][ref_pos][prev_state]
        forward[i, j, k, l, 1] = result
        # print "read: ", i, "len: ", k, "ref: ", j, "len: ", l, states[1], forward[i, j, k, l, 1]
        # print i, j, k, l, states[1], forward[i, j, k, l, 1]
        return 0

    # start from 0, because there can be insertions and deletions at the begining
    for i in xrange(len(read)):

        for j in xrange(len(reference)):
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
                forward_position[i][j][state] = by_iter_slog(numpy.nditer(forward[i, j, :, :, state]))
                #if not math.isnan(forward_position[i][j][state]):
                #    print "---------------------------", i, j, states[state], forward_position[i][j][state]
    check = by_iter_slog(numpy.nditer(forward_position[len(read_tmp), len(reference) - 1, :]))
    # print check
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
    reference = [homopolymer()]
    reference.extend(nucl_to_hp(reference_tmp))
    len_ref = len(reference) - 1  # true length
    len_read = len(read_tmp)
    # print "Len_read: ", len_read, "len_reference: ", len_ref
    # position and create len_sequence to remember true length
    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion', 3: 'Begin', 4: 'End'}
    state_index = {'Match': 0, 'Deletion': 1, 'Insertion': 2, 'Begin': 3, 'End': 4}
    max_hp_read_s = [0] + len_max_hp_start(read_tmp)  # because read start from ' '

    # information about the lengthest HP, ending from this position
    max_hp_read_e = [0] + len_max_hp_end(read_tmp)  # because read start from ' '
    max_hp_ref_e = [0] + len_max_hp_end(reference_tmp)  # because reference start from ' '

    backward = float("-inf")*numpy.ones(shape=[len_read + 1, len_ref + 1, max(max_hp_read_e) + 1,
                                               max(max_hp_ref_e) + 1,  5], dtype=float)
    # print "Size: ", backward.nbytes
    # print "Shape: ", backward.shape

    # initialize
    # B(i, m, 0, 1, Deletion)
    backward[len_read, len_ref, 0, reference[len_ref].length, state_index['Deletion']] = 0
    # B(n, m, k, 0, Insertion)
    emiss = [model.HMM[len_ref].emission(homopolymer(), homopolymer(read[len_read], t), 'Insertion')
             for t in range(1, length_last_hp(read) + 1)]
    backward[len_read, len_ref, 1: length_last_hp(read) + 1, 0, state_index['Insertion']] = emiss
    # B(n, m, k, 1, Match)
    emiss = [model.HMM[len_ref].emission(reference[len_ref], homopolymer(read[len_read], t), 'Match')
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
        emiss = [model.HMM[j + 1].emission(hp_input, homopolymer(read_base, t), 'Match') for t in range(1, hp_len + 1)]
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
        emiss = [model.HMM[j].emission(hp_input, homopolymer(read_base, t), 'Insertion') for t in range(1, hp_len + 1)]
        bck = [backward[i + t, j, t, 0, st_index] for t in range(1, hp_len + 1)]
        # print 'Insertion', i + hp_len, j, hp_len, reference[j].length, backward[i + hp_len, j, hp_len, reference[j].length, st_index]
        result = [log_product(x, y) for x, y in zip(emiss, bck)]
        if len(result) == 1:
            return result[0]
        else:
            return iter_slog(result)


    for i in xrange(len_read, -1, -1):  # read position

        for j in xrange(len_ref, -1, -1):     # reference position
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
                trans_prob = [model.HMM[j].transition('Match', states[k]) for k in range(len(states))]
                if i != len_read and read[i] == read[i + 1]:
                    trans_prob[2] = float("-inf")
                value_1 = [log_product(x, y) for x, y in zip(trans_prob, part_two)]
                value = iter_slog(value_1)
            for k in range(1, max_hp_read_e[i] + 1):
                backward[i, j, k, reference[j].length, state_index['Match']] = value
            # if i < 5 and j < 5:
            #     print "---------------------------", i, j, 'Match', "{0:.2f}".format(value),
            # [round(x, 3) for x in part_two], [round(x, 3) for x in
            # [model.HMM[j].transition('Match', states[k]) for k in range(len(states))]]

            # Count B(i, j, k, l, Deletion)
            trans_prob = [model.HMM[j].transition('Deletion', states[k]) for k in range(len(states))]
            value = [log_product(x, y) for x, y in zip(trans_prob, part_two)]
            value = iter_slog(value)
            # if i < 5 and j < 5:
            #     print "---------------------------", i, j, 'Deletion', "{0:.2f}".format(value),
            # [round(x, 3) for x in part_two], [round(x, 3) for x in trans_prob]
            backward[i, j, 0,  reference[j].length, state_index['Deletion']] = value

            # Count B(i, j, k, l, Insertion)
            trans_prob = [model.HMM[j].transition('Insertion', states[k]) for k in range(len(states))]
            if i != len_read and read[i] == read[i + 1]:
                trans_prob[0] = float("-inf")
            value = [log_product(x, y) for x, y in zip(trans_prob, part_two)]
            value = iter_slog(value)
            # if i < 5 and j < 5:
            #     print "---------------------------", i, j, 'Insertion', "{0:.2f}".format(value), [round(x, 3) for x in part_two], [round(x, 3) for x in trans_prob]
            for k in range(1, max_hp_read_e[i] + 1):
                backward[i, j, k, 0, state_index['Insertion']] = value
    check = by_iter_slog(numpy.nditer(backward[0, 0, :, :, :]))
    # print check
    return backward, check


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
    reference = [homopolymer()]
    reference.extend(nucl_to_hp(reference_tmp))

    max_hp_read = [0] + len_max_hp_end(read_tmp) + [0]    # information about the lengthest HP at each position
    max_hp_ref = [0] + len_max_hp_end(reference_tmp) + [0]
    ref_hmm_blocks = hmm_block(reference)     # information about HMM blocks

    forward, check_forward = count_forward(read_tmp, reference_tmp, model)
    backward, check_backward = count_backward(read_tmp, reference_tmp, model)
    #  check_forward and check_backward must be same
    print "Difference: ", abs(check_forward - check_backward)
    # shape of forward and backward must also be the same
    # print "Shape: ", forward.shape, backward.shape

    # count first missing variable. Amazing :)
    gamma = forward + backward
    gamma -= check_forward
    # print "Shape of gamma: ", gamma.shape

    # count second missing variable. Need to do it faster - by presenting operations as a vector operations
    # xi(i, j, pi, pi', k, l)
    xi = float("-inf")*numpy.ones(shape=[len(read), len(reference), 5, 5, max(max_hp_read) + 1, max(max_hp_ref) + 1],
                                        dtype=float)

    def supp(curr_st, prev_st, i, j, k, l, hp_input, hp_output, check):
        # print "transition", model.HMM[j].transition(states[prev_st], states[curr_st])
        transition = model.HMM[j].transition(states[prev_st], states[curr_st])
        # print "emission", model.HMM[j].emission(hp_input, hp_output, states[curr_st]), hp_input.base, hp_output.base, states[curr_st], j
        emission = model.HMM[j].emission(hp_input, hp_output, states[curr_st])
        backward_curr = backward[i, j, k, l, curr_st]
        it = numpy.array([[forward[i - k, j - l, k_tmp, l_tmp, prev_state]
               for k_tmp in range(max_hp_read[i - k] + 1)] for l_tmp in range(2)])
        it = it.flatten()
        it = iter_slog(it)
        xi[i, j, curr_st, prev_st, k, l] = iter_plog([transition, backward_curr, emission,  it, (-1)*check])
        return 0

    for i in xrange(1, len(read)):  # read position, start from 1 (count position from 1)
        for j in xrange(1, len(reference)):     # reference position
            for k in range(max_hp_read[i] + 1):     # maximum length of HP, ending at i
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
                    # transition_matrix[prev_state, curr_state] = log_sum(transition_matrix[prev_state, curr_state], xi[i, j, curr_state, prev_state, k, l])
    return gamma, xi


def update_parameters(training_set, base_model, max_hp_len, b, sigma, hp_freq):
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
        for i in range(len(states)):
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

    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion', 3: 'Begin', 4: 'End'}
    bases = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    transition_matrix = float("-inf")*numpy.ones(shape=[len(states), len(states)], dtype=float)
    length_call_matrix = float("-inf")*numpy.ones(shape=[max_hp_len + 1, max_hp_len + 1], dtype=float)
    ins_base_call = float("-inf")*numpy.ones(shape=[4], dtype=float)
    counter = 0
    for pair in training_set:   # process every pair of read, reference
        print counter, "Step", '\n'
        if counter == 6:
            break
        counter += 1

        read = pair[0][0:20]
        reference = pair[1][0:20]
        # xi shape: [read_pos, ref_pos, prev_state, curr_state, hp_read_len, hp_ref_len]
        # gamma shape: [read_pos, ref_pos, hp_read_len, hp_ref_len, state]
        gamma, xi = count_missing_variables(base_model, read, reference)

        max_hp_read = max(len_max_hp_end(read))    # information about the lengthest HP
        max_hp_ref = max(len_max_hp_end(reference))

        # update T (transition matrix), T(pi, pi')
        # !!! Here I don't use begin and end
        for previous in states:
            for current in states:
                transition_matrix[previous, current] = by_iter_slog(numpy.nditer(xi[:, :, previous, current, :, :]))

        # update L(k,l) - occurrences of length calling
        for k in xrange(max_hp_read):     # maximum length of HP. All of unuseful will be -inf
            for l in xrange(max_hp_ref):
                length_call_matrix[l, k] = by_iter_slog(numpy.nditer(gamma[:, :, k, l, :]))

        # update length call for insertion. gamma[i + 1...], because here count read from 0, in gamma from 1
        for i in xrange(len(read)):
            ins_base_call[bases[read[i]]] = log_sum(ins_base_call[bases[read[i]]],
                                                    by_iter_slog(numpy.nditer(gamma[i + 1, :, 1:, 0, 2])))
        gamma = xi = None

    # normalize
    print "Transition matrix: "
    transition_matrix = transition_normalize(transition_matrix)
    for i in range(len(states)):
        print [round(numpy.exp(transition_matrix[i, j]), 4) for j in range(len(transition_matrix[i, ]))]

    ins_base_call = base_call_normalize(ins_base_call)
    print "Insertion base call: ", [round(numpy.exp(ins_base_call[i]), 4) for i in range(len(ins_base_call))]
    print "Sigma", sigma
    length_call_match, length_call_ins, b, sigma = update_length_call_parameters(length_call_matrix,
                                                                                       b, max_hp_len, hp_freq, sigma)
    print "670, Transition matrix", '\n'
    for i in range(len(states)):
        print [round(numpy.exp(transition_matrix[i, j]), 4) for j in range(len(transition_matrix[i, ]))]
    print "Insertion base call: ", [round(numpy.exp(ins_base_call[i]), 4) for i in range(len(ins_base_call))]

    print "Insertion length call: ", [round(numpy.exp(length_call_ins[i]), 4) for i in range(len(length_call_ins))]
    for i in range(len(length_call_match[1,])):
        print [round(numpy.exp(length_call_match[i, j]), 4) for j in range(len(length_call_match[i, ]))]
    print "Length call match", '\n', length_call_match[1,]

    # new_model = hmm.HmmModel(ins_base_call, length_call_match, length_call_ins, transition_matrix)
    return 0


def update_length_call_parameters(length_call_matrix_ln, b, max_length_hp, p_k, sigma):
    """
    :param length_call_matrix: matrix with log-probabilities
    :param b: one-dimensional list with parameters of Laplace distrib
    :param max_length_hp: maximum length of HP
    :param p_k: homopolymers length array (observed from genome)
    :param sigma: parameter of log-normal distribution
    :return: udpated length call matrix, length cal insertion array
    """
    length_call_matrix = numpy.exp(length_call_matrix_ln)
    print length_call_matrix
    f_start = 0
    f_end = max_length_hp + 1
    f = numpy.arange(f_start, f_end, 0.01)

    def count_p_f_l(b_scale):
        """
        Count p(f|l) - Laplace distribution, probability of flow intensity f, when input HP have length l
        :param b: parameter of scale
        :return: 0
        """
        tmp = numpy.zeros(shape=[len(f), max_length_hp + 1], dtype=float)
        for ff in xrange(len(f)):
            for l in range(1, max_length_hp + 1):   # l = 0 count by log-normal distribution
                tmp[ff, l] = laplace.pdf(f[ff] , loc=l, scale=b_scale[l])
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
        # for k in range(1, max_length_hp + 1):
        #     for ff in xrange(len(f)):
        #         tmp[k, ff] = p_f_l[ff, k] * p_k[k] / z_f[ff]
        return tmp

    def count_p_f_zero(sigma):
        tmp = []
        print f
        for ff in range(len(f)):
            tmp.append(lognorm.pdf(f[ff], 1, loc=0, scale=sigma))
        return tmp

    def expectation_step():
        """
        Count p(f|k,l) (p_f_k_l and p_f_k_zero) (See supplementary)
        """
        tmp = numpy.zeros(shape=[len(f), max_length_hp + 1, max_length_hp + 1], dtype=float)     # p(f | k, l)
        tmp_zero = numpy.zeros(shape=[len(f), max_length_hp + 1], dtype=float)     # p(f | k, 0)
        for k in range(1, max_length_hp + 1):
            tst = [p_k_f[k, i] * p_f_zero[i] for i in range(len(f))]
            if sum(tst) != 0:
                tst = tst / sum(tst)
            tmp_zero[:, k] = tst[:]
            for l in range(1, max_length_hp + 1):
                tst = [p_k_f[k, i]*p_f_l[i, l] for i in range(len(f))]
                if sum(tst) != 0:
                    tst = tst / sum(tst)
                tmp[:, k, l] = tst[:]  # p(k|f) * p(f|l)
        return tmp_zero, tmp

    def prod(x, y):
            assert (len(x) == len(y)), "773"
            res = 0
            for i in range(len(x)):
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
        for l in range(1, max_length_hp + 1):
            numerator = 0
            denominator = 0
            for k in range(1, max_length_hp + 1):
                # print '\n', l, k, [round(p_f_k_l[i, k, l], 4) for i in range(len(p_f_k_l[:, k, l])) if round(p_f_k_l[i, k, l], 4) != 0.0]
                # print length_call_matrix[l, k]
                tmp_num = [p_f_k_l[ff, k, l] * abs(f[ff] - l) for ff in xrange(len(f))]
                numerator = numerator + length_call_matrix[l, k] * sum(tmp_num)
                denominator = denominator + sum(p_f_k_l[:, k, l]) * length_call_matrix[l, k]
            # for ff in xrange(f):
            #     for k in range(1, max_length_hp + 1):
            #         temp = p_f_k_l[ff, k, l]*length_call_matrix[k, l]
            #         numerator += temp * abs(ff - l)
            #         denominator += temp
            print numerator, denominator
            tmp_b[l] = numerator / denominator
        return tmp_b

    def count_sigma():
        numerator = sum([prod(p_f_k_zero[ff, :], length_call_matrix[0, :])*eln(f[ff])**2 for ff in xrange(len(f))])
        denominator = sum([prod(p_f_k_zero[ff, :], length_call_matrix[0, :]) for ff in xrange(len(f))])
        return math.sqrt(numerator / denominator)

    def count_length_call_match(max_hp_length, scale):
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
            num = laplace.pdf(x, loc=l, scale=scale[l]) * laplace.pdf(x, loc=k, scale=scale[k]) * p_k[k]
            denom = sum([p_k[i] * laplace.pdf(x, loc=i, scale=scale[i]) for i in range(1, max_hp_length + 1)])
            return num/denom

        def normalize(item, max_len):
            for i in range(max_len):
                item[i, ] = item[i, ] / sum(item[i, ])
            return item

        for l in range(1, max_hp_length + 1):
            for k in range(1, max_hp_length + 1):
                result[l - 1, k - 1] = quad(lcall, 0, max_hp_length, args=(l, k))[0]
        result = normalize(result, max_hp_length)
        return result

    def count_length_insertion(max_hp_length, sigma_scale, b_scale):
        """
        Count length call in case zero-length of input hp
        :param max_hp_length: maximum hp length
        :param sigma_scale: scale parameter for log-normal distribution
        :param b_scale: scale parameters (an array) for Laplace distribution
        :return:
        """
        result = numpy.zeros(shape=[max_hp_length], dtype=float)

        def lcall(x, k):
            num = lognorm.pdf(x, 1, loc=0, scale=sigma_scale) * laplace.pdf(x, loc=k, scale=b_scale[k])
            denom = sum([p_k[i] * laplace.pdf(x, loc=i, scale=b_scale[i]) for i in range(1, max_hp_length + 1)])
            return num/denom

        for k in range(1, max_hp_length + 1):
            result[k - 1] = quad(lcall, 0, max_hp_length, args=(k))[0]
        result = result / sum(result)
        return result

    p_f_l = count_p_f_l(b)     # p(f | l)
    z_f = count_z_f()      # Coefficient of normalize Z = \sum_{k}p(f | k) * p(k)
    p_k_f = count_p_k_f()       # p(k | f)
    p_f_zero = count_p_f_zero(sigma)    # p(f | 0)
    p_f_k_zero, p_f_k_l = expectation_step()     # p(f | k, 0),  p(f | k, l)

    """
    print "\n p(f | l)"
    for i in range(0, len(f), 10):
        print [round(p_f_l[i, j], 4) for j in range(12)]

    print "\n Z_f", [round(z_f[i], 4) for i in range(10)]

    print "\n p(k | f)"
    for i in range(10):
        print [round(p_k_f[i, j], 4) for j in range(0, len(f), 5)]

    print "\n p(f | 0)", [round(p_f_zero[i], 4) for i in range(len(f))]

    print "\n p(f | k, 0)"
    for i in range(0, len(f), 100):
        print [round(p_f_k_zero[i, j], 4) for j in range(10)]

    print "\n p(f | k, l)"
    for ff in range(100, 1299, 100):
        print "\n p(f | k, l)", ff
        for k in range(10):
            print [round(p_f_k_l[ff, k, l], 4) for l in range(10)]
    """

    new_b = count_b()
    print "old b: ", b
    print "new b: ", new_b
    new_sigma = count_sigma()
    print "Sigma: ", sigma, new_sigma

    length_call_match = count_length_call_match(max_length_hp, new_b)
    length_call_insertion = count_length_insertion(max_length_hp, new_sigma, new_b)
    return length_call_match, length_call_insertion, new_b, new_sigma


def main():
    """
    Implement training of HMM.
    :param sample_data: {(r^d, t^d), d = 1..N} - N pair of reads and mapped reference segments
    :param model: start model
    :return:
    """
    hmm_test = hmm.HmmModel()
    """
    read = "GCGTTTGGCGTCGAACCCAATTCCCGCCTCATTGGAAAACATACTGCGCCCAAATGACGTGGGGAAGTTGCCCGATATTCATTACG"
    reference = "GCGTTTGGCGTCGAACCCATTCCCGCCTCATTGGAAAACATACTGCGCTGAAAACCGTTAGTAATCGCCTGGCTTAAGGTA"
    print "Count forward"
    count_forward(read, reference, hmm_test)
    print "Count backward"
    count_backward(read, reference, hmm_test)
    print "Count missing variables"
    count_missing_variables(hmm_test, read, reference)
    """

    train_set, freq, b = form_dataset()
    print "Dataset formed"
    update_parameters(train_set, hmm_test, 12, b, 4, freq)

    return 0



main()