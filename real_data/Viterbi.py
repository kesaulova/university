import numpy
import re
import addmath
import hmm

discrete_distribution = addmath.discrete_distribution
eln = addmath.eln
exp = addmath.eexp
log_product = addmath.log_product
iter_plog = addmath.iter_plog

states = {0: 'Match', 1: 'Deletion', 2: 'Insertion', 3: 'Begin', 4: 'End'}

def process_read(read):
    """
    Create string were at each positions is length of lengthiest hmm.homopolymer, ended at these position
    :param read: nucleotide sequence
    :return: list with numbers, indicated length of the longest hmm.hmm.homopolymer, ended at these position
    """
    result = [1]*len(read)
    base = read[0]
    for i in xrange(1, len(read)):
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
    :return: List with four elements (index of previous element)
    """
    # print index_string
    if index_string == 'Begin':
        return [0, 0, 0, 0]
    else:
        result = [int(item) for item in re.split(' ', index_string)]
        return result


def length_first_hp(read):
    """
    :param read: nucleotide sequence
    :return: length of first homopolymer in read
    """
    result = 1
    base = read[1]
    for i in xrange(2, len(read)):
        if read[i] == base:
            result += 1
        else:
            break
    return result


def length_last_hp(read):
    """
    Detect length of last homopolymer at sequence
    :param read: nucleotide sequence
    :return: length of last HP
    """
    result = 1
    base = read[len(read) - 1]
    for i in xrange(len(read) - 2, -1, -1):
        if read[i] == base:
            result += 1
        else:
            break
    return result


def viterbi_initialize(model, reference, read, k_max, viterbi_probability,  viterbi_backtracking):
    """
    Initialize viterbi variables.
    All arrays fill with -inf, maybe I should delete half of these code.
    :param model: HMM model
    :param reference: reference as homopolymer sequence
    :param read: read as nucleotide sequence
    :param k_max: array, containing for each position length of lengthiest HP, ending at this position
    :param viterbi_probability: probability array
    :param viterbi_backtracking: array, containing indexes of previous elements in path
    :return:
    """
    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion', 3: 'Begin'}#, 4: 'End'}
    # V(0,j,0,M) = V(0,j,0,I) = 0, V(0,j,0,D) != 0
    viterbi_probability[0, 0, 0, 0] = float("-inf")
    viterbi_backtracking[0, 0, 0, 0] = 'Impossible'
    viterbi_probability[0, 0, 0, 1] = float("-inf")
    viterbi_backtracking[0, 0, 0, 1] = 'Impossible'
    viterbi_probability[0, 0, 0, 2] = float("-inf")
    viterbi_backtracking[0, 0, 0, 2] = 'Impossible'
    viterbi_probability[0, 0, 0, 4] = float("-inf")
    viterbi_backtracking[0, 0, 0, 4] = 'Impossible'

    # V(0,0,k,pi) = 0
    for k in xrange(max(k_max)):
        for state in states:
            viterbi_backtracking[0, 0, k, state] = 'Impossible'
            viterbi_probability[0, 0, k, state] = float("-inf")
    # V(i,j,k,Begin) = 0 except V(0,0,0,Begin) ( =1 )
    for i in xrange(len(read)):
        for j in xrange(len(reference)):
            for k in xrange(max(k_max)):
                viterbi_backtracking[i, j, k] = 'Impossible'

    for j in xrange(1, len(reference)):
        viterbi_probability[0, j, 0, 0] = float("-inf")
        viterbi_backtracking[0, j, 0, 0] = 'Impossible'
        viterbi_probability[0, j, 0, 2] = float("-inf")
        viterbi_backtracking[0, j, 0, 2] = 'Impossible'
        if j == 1:
            viterbi_probability[0, j, 0, 1] = model.HMM[0].transition('Begin', 'Deletion')
            # print 0, j, 0, states[1], viterbi_probability[0][j][0][1]
            viterbi_backtracking[0, j, 0, 1] = 'Begin'
        else:
            viterbi_probability[0, j, 0, 1] = log_product(model.HMM[j].transition('Deletion', 'Deletion'),
                                                          viterbi_probability[0][j - 1][0][1])
            # print 0, j, 0, states[1], viterbi_probability[0][j][0][1]
            viterbi_backtracking[0, j, 0, 1] = str(0) + ' ' + str(j - 1) + ' ' + str(0) + ' ' + str(1)
    # V(i,0,k,M) = V(i,0,k,D) = 0
    for i in xrange(1, len(read)):

        # k count from 1 - it is length of HP in sequence
        for k in xrange(1, k_max[i] + 1):
            viterbi_probability[i, 0, k, 0] = float("-inf")     # V(i,0,k,Match)
            viterbi_backtracking[i, 0, k, 0] = 'Impossible'

            viterbi_probability[i, 0, k, 1] = float("-inf")    # V(i,0,k,Deletion)
            viterbi_backtracking[i, 0, k, 1] = 'Impossible'

            # Count V(i,0,k,Insertion). Have to find max among V(i - k, 0, 1:k_max[i - k], Insertion)
            # 1:k_max[i - k], because there is no deletions
            # First, initialize V(i,0,k,I) when i < k_max[(len(first hmm.homopolymer)], because in this case transition
            # probabilities is initial probabilities

            if k == i and k <= length_first_hp(read):     # case V(i, 0, i, Ins) - come from begining, i = len(firstHP)
                trans_prob = model.HMM[0].transition('Begin', 'Insertion')
                current_hp = hmm.homopolymer(read[i], k)
                emiss_prob = model.HMM[0].emission(hmm.homopolymer(), current_hp, 'Insertion')
                viterbi_probability[i, 0, k, 2] = log_product(trans_prob, emiss_prob)
                viterbi_backtracking[i, 0, k, 2] = 'Begin'
                # print "---------------------------------- line 140", i, 0, k, states[2], 'Prob: ', \
                #     viterbi_probability[i, 0, k, 2]
                continue
            max_prob = float("-inf")
            number = [0, 0, 0, 0]
            current_hp = hmm.homopolymer(read[i], k)
            value = float("-inf")
            for k_prev in xrange(1, k_max[i - k] + 1):
                for state in states:
                    # print "Previous, line 126 ", i - k, 0, k_prev, states[state], "transition ", 
                    # model.HMM[0].transition(states[state], 'Insertion'), \
                    # "viterbi: ", viterbi_probability[i - k, 0, k_prev, state]
                    # value = tran_prob * V(previous)
                    value = log_product(model.HMM[0].transition(states[state], 'Insertion'),
                                        viterbi_probability[i - k, 0, k_prev, state])
                    # print value
                    # Have to find max among V(i - k, 0, 1:k_max[i - k], Insertion)
                    if value > max_prob:
                        max_prob = value
                        number = [i - k, 0, k_prev, state]
                    # count probability

            assert (k <= 10), "k is too large: " + str(k)
            emiss_prob = model.HMM[0].emission(hmm.homopolymer(), current_hp, 'Insertion')
            viterbi_probability[i, 0, k, 2] = log_product(value, emiss_prob)
            viterbi_backtracking[i, 0, k, 2] = str(number[0]) + ' ' + str(number[1]) + ' ' + str(number[2]) + \
                                                   ' ' + str(number[3])
            # print "---------------------------------- line 140", i, 0, k, states[2], 'Prob: ', \
            # viterbi_probability[i, 0, k, 2]
        viterbi_probability[0, 0, 0, 3] = 0
        viterbi_backtracking[0, 0, 0, 3] = 'Begin'
    return viterbi_probability, viterbi_backtracking


def process_match(i, j, k, read, reference, k_max, viterbi_probability, viterbi_backtracking, model):
    """
    Process 'Match' case: V(i, j, k, M) = max(k',pi')(V(i - k, j - 1, k', pi')*p(M|pi')*emission
    :param i: read position
    :param j: reference HP position, = number of ceurrent block in HMM
    :param k: length of output HP
    :param read: read as nucleotide sequence
    :param reference: reference as homopolymer sequence
    :param k_max: array, containing for each position length of lengthiest HP, ending at this position
    :param viterbi_probability: probability array
    :param viterbi_backtracking: array, containing indexes of previous elements in path
    :param model: HMM model
    :return:
    """
    if read[i] != reference[j].base:    # in Match bases should be the same, not the different
        viterbi_backtracking[i, j, k, 0] = 'No sense'
        return 0

    max_prob = float("-inf")
    prev_index = [0]*4
    emiss_prob = model.HMM[j].emission(reference[j], hmm.homopolymer(read[i], k), 'Match')
    # we want to find most probable path until this moment
    for k_prev in xrange(0, k_max[i - k] + 1):
        trans_prob = [model.HMM[j - 1].transition(states[state], 'Match') for state in states]
        val = viterbi_probability[i - k, j - 1, k_prev, :] + trans_prob
        val_max = numpy.argmax(val)
        if val[val_max] > max_prob:
            max_prob = val[val_max]
            prev_index = [i - k, j - 1, k_prev, val_max]
    viterbi_backtracking[i, j, k, 0] = str(prev_index[0]) + ' ' + str(prev_index[1]) + ' ' + \
                                       str(prev_index[2]) + ' ' + str(prev_index[3])
    viterbi_probability[i, j, k, 0] = log_product(max_prob, emiss_prob)
    return 0


def process_insertion(i, j, k, read, reference, k_max, viterbi_probability, viterbi_backtracking, model):
    """
    Process 'Insertion' case: V(i, j, k, I) = max(k',pi')(V(i - k, j, k', pi')*p(I|pi')*emission
    :param i: read position
    :param j: reference HP position, = number of ceurrent block in HMM
    :param k: length of output HP
    :param read: read as nucleotide sequence
    :param reference: reference as homopolymer sequence
    :param k_max: array, containing for each position length of lengthiest HP, ending at this position
    :param viterbi_probability: probability array
    :param viterbi_backtracking: array, containing indexes of previous elements in path
    :param model: HMM model
    :return:
    """

    max_prob = float("-inf")
    prev_index = [0]*4
    emiss_prob = model.HMM[j].emission(hmm.homopolymer(), hmm.homopolymer(read[i], k), 'Insertion')
    # value = float("-inf")

    for k_prev in xrange(k_max[i - k] + 1):
        trans_prob = [model.HMM[j].transition(states[state], 'Insertion') for state in states]
        val = viterbi_probability[i - k, j, k_prev, :] + trans_prob
        val_max = numpy.argmax(val)
        if val[val_max] > max_prob:
            max_prob = val[val_max]
            prev_index = [i - k, j, k_prev, val_max]

    viterbi_probability[i, j, k, 2] = log_product(max_prob, emiss_prob)
    viterbi_backtracking[i, j, k, 2] = str(prev_index[0]) + ' ' + str(prev_index[1]) + ' ' + str(prev_index[2]) + \
                                       ' ' + str(prev_index[3])
    return 0


def process_deletion(i, j, k, viterbi_probability, viterbi_backtracking, model):
    """
    Count V(i, j, k, Deletion), process 'Deletion' case:
    V(i, j, k, I) = max(pi')(V(i, j - 1, k', pi')*p(D|pi')
    :param i: read position
    :param j: reference HP position, = number of ceurrent block in HMM
    :param k: length of output HP
    :param viterbi_probability: probability array
    :param viterbi_backtracking: array, containig indexes of previous elements in path
    :param model: HMM model
    :return:
    """
    max_prob = float("-inf")
    prev_index = [0]*4
    # value = float("-inf")
    trans_prob = [model.HMM[j - 1].transition(states[state], 'Deletion') for state in states]
    val = viterbi_probability[i, j - 1, k, :] + trans_prob
    val_max = numpy.argmax(val)
    if val[val_max] > max_prob:
        max_prob = val[val_max]
        prev_index = [i, j - 1, k, val_max]

    viterbi_probability[i, j, k, 1] = max_prob
    viterbi_backtracking[i, j, k, 1] = str(prev_index[0]) + ' ' + str(prev_index[1]) + ' ' + str(prev_index[2]) + \
                                       ' ' + str(prev_index[3])
    return 0


def parse_viterbi(probability, backtracking, read, reference):
    """
    Detect path of hidden state, when probability and backtracking array filled.
    :param probability: array of probabilities of Viterbi variables
    :param backtracking: array of previous indexes in paths for Viterbi variables
    :param read: nucleotide sequence of read
    :param reference: ...
    :return: sequence of hidden state, sequence of probability
    """
    reference_length = len(reference)
    read_length = len(read)
    last_hp = length_last_hp(read)
    path_probability = 0

    # get number of state with max prob
    state_max = numpy.argmax(probability[read_length, reference_length, last_hp, :3])
    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion', 3: 'Begin'}
    path = [states[state_max]]
    inf = parse_backtracking(backtracking[read_length, reference_length, last_hp, state_max])
    path.append(states[inf[3]])
    while backtracking[inf[0], inf[1], inf[2], inf[3]] != 'Begin':
        path_probability += probability[inf[0], inf[1], inf[2], inf[3]]
        inf = parse_backtracking(backtracking[inf[0], inf[1], inf[2], inf[3]])
        path.append(states[inf[3]])
    # return path[::-1], path_probability
    return path[::-1], probability[read_length, reference_length, last_hp, state_max]

def viterbi_path(read, reference, model):
    """
    :param read: nucleotide sequence
    :param reference: nucleotide sequence
    :return: most probable path of hidden state and its probability
    """


    k_max = [0] + process_read(read)    # information about length of lengthiest HP's, ended at each positions
    read = ' ' + read
    reference = [hmm.homopolymer()] + hmm.nucleotide_to_homopolymer(reference)
    max_k_value = max(k_max) + 1
    viterbi_probability = float("-inf")*numpy.ones(shape=[len(read), len(reference), max_k_value, 5], dtype=float)
    viterbi_backtracking = numpy.ones(shape=[len(read), len(reference), max_k_value, 5], dtype=basestring)
    viterbi_probability, viterbi_backtracking = viterbi_initialize(model, reference, read, k_max, viterbi_probability,
                                                                   viterbi_backtracking)

    for i in xrange(1, len(read)):
        for j in xrange(1, len(reference)):
            for k in xrange(1, k_max[i] + 1):
                process_match(i, j, k, read, reference, k_max, viterbi_probability, viterbi_backtracking, model)
                process_insertion(i, j, k, read, reference, k_max, viterbi_probability, viterbi_backtracking, model)
                process_deletion(i, j, k, viterbi_probability, viterbi_backtracking, model)
    path, prob = parse_viterbi(viterbi_probability, viterbi_backtracking, read[1:], reference[1:])

    return path, prob

#
# rf = "CCGAACGTATTTTGCCGA"
# rd = "CCGAACGTATTTTGCCGA"
# md = hmm.HmmModel()
# print viterbi_path(rd, rf, md)[0]
# print viterbi_path(rd, rf, md)[1]
#
# rf_nucl = hmm.nucleotide_to_homopolymer(rf)
# rd_nucl = hmm.nucleotide_to_homopolymer(rd)
# res = 0
#
# for i in range(len(rf_nucl)):
#     res += md.HMM[1].emission(rf_nucl[i], rd_nucl[i], 'Match')
# print res + numpy.log(0.9)*len(rf_nucl)

