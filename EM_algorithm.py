import hmm
import numpy
import addmath

eln = addmath.eln
exp = addmath.eexp
log_product = addmath.log_product
log_sum = addmath.log_sum
iter_plog = addmath.iter_plog
iter_slog = addmath.iter_slog
homopolymer = hmm.homopolymer


def length_last_hp(read):
    result = 1
    base = read[len(read) - 1]
    for i in range(len(read) - 2, -1, -1):
        if read[i] == base:
            result += 1
        else:
            break
    return result


def process_sequence_end(sequence):
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


def process_sequence_start(sequence):
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


# I count F and B from 1! Read and reference
def count_forward(read_tmp, reference_tmp, model):
    read = ' ' + read_tmp
    reference = ' ' + reference_tmp
    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion', 3: 'Begin', 4: 'End'}
    max_hp_read = [0]
    max_hp_read.extend(process_sequence_end(read))    # information about the lengthest HP at each position
    max_hp_read.append(0)
    max_hp_ref = [0]
    max_hp_ref.extend(process_sequence_end(reference))
    max_hp_ref.append(0)
    ref_hmm_blocks = hmm_block(reference)     # information about HMM blocks
    forward = float("-inf")*numpy.ones(shape=[len(read), len(reference), 5, max(max_hp_read) + 1, max(max_hp_ref) + 1], dtype=float)
    print forward.shape
    forward_position = float("-inf")*numpy.ones(shape=[len(read), len(reference), 5], dtype=float)
    forward_position[0][0][3] = 0

    def process_match(i, j, k, l, hp_input, hp_output):
        """
        Count F(i, j, k, l, Match)
        :param i: read position
        :param j: reference position
        :param k: length of HP, ended at position i
        :param l: length of HP, ended at position j
        :param hp_input: HP in ref
        :param hp_output: HP in read
        :return:
        """
        temp_result = float("-inf")
        if i == 0:
            read_pos = 0
        else:
            read_pos = i - k
        if j == 0:
            ref_pos = 0
        else:
            ref_pos = j - l
        for prev_state in states:
            # in Match we can go only from previous block. But... It is not exactly previous block...
            transition = eln(model.HMM[ref_hmm_blocks[ref_pos]].transition(states[prev_state], 'Match'))
            emission = eln(model.HMM[ref_hmm_blocks[j]].emission(hp_input, hp_output, 'Match'))
            temp_result = log_sum(temp_result, iter_plog([forward_position[read_pos][ref_pos][prev_state], transition, emission]))
            # print states[prev_state], read_pos, ref_pos, prev_state, k, l, forward[read_pos][ref_pos][prev_state][k][l]
        forward[i][j][0][k][l] = temp_result
        # print "read: ", i, "len: ", k, "ref: ", j, "len: ", l, states[0], forward[i][j][0][k][l]
        print i, j, k, l, states[0], forward[i][j][0][k][l]
        return 0

    def process_insertion(i, j, k, l, hp_output):
        temp_result = float("-inf")
        if i == 0:
            read_pos = 0
        else:
            read_pos = i - k
        for prev_state in states:
            transition = eln(model.HMM[ref_hmm_blocks[j]].transition(states[prev_state], 'Insertion'))
            emission = eln(model.HMM[ref_hmm_blocks[j]].emission(homopolymer(), hp_output, 'Insertion'))
            temp_result = log_sum(temp_result, iter_plog([forward_position[read_pos][j - 1][prev_state], transition, emission]))
        forward[i][j][2][k][l] = temp_result
        # print "read: ", i, "len: ", k, "ref: ", j, "len: ", l, states[2], forward[i][j][2][k][l]
        print i, j, k, l, states[2], forward[i][j][2][k][l]
        return 0

    def process_deletion(i, j, k, l, hp_input):
        temp_result = float("-inf")
        if j == 0:
            ref_pos = 0
        else:
            ref_pos = j - l
        if i == 0:
            read_pos = 0
        else:
            read_pos = i - 1
        for prev_state in states:
            transition = eln(model.HMM[ref_hmm_blocks[ref_pos]].transition(states[prev_state], 'Deletion'))
            emission = eln(model.HMM[ref_hmm_blocks[j]].emission(hp_input, homopolymer(), 'Deletion'))
            temp_result = log_sum(temp_result, iter_plog([forward_position[read_pos][ref_pos][prev_state], transition, emission]))
            #print states[prev_state], read_pos, ref_pos, prev_state, k, l, forward_position[read_pos][ref_pos][prev_state]
        forward[i][j][1][k][l] = temp_result
        #print "read: ", i, "len: ", k, "ref: ", j, "len: ", l, states[1], forward[i][j][1][k][l]
        print i, j, k, l, states[1], forward[i][j][1][k][l]
        return 0

    for i in range(len(read)):

        for j in range(len(reference)):

            for k in range(max_hp_read[i] + 1):

                for l in range(max_hp_ref[j] + 1):
                    print "169, ", i, j, k, l
                    if l == 0:
                        hp_input = homopolymer()
                    else:
                        hp_input = homopolymer(reference[j], l)

                    if k == 0:
                        hp_output = homopolymer()
                    else:
                        hp_output = homopolymer(read[i], k)

                    if hp_input.base == '-' and hp_output.base == '-':
                        continue
                    elif hp_output.base == '-':
                        process_deletion(i, j, k, l, hp_input)
                    elif hp_input.base == '-':
                        process_insertion(i, j, k, l, hp_output)
                    elif hp_input.base == hp_output.base:
                        process_match(i, j, k, l, hp_input, hp_output)

            # fill F(i,j,pi)
            for state in states:
                forward_temp = forward_position[i][j][state]
                for t in range(max(max_hp_read)):
                    forward_temp = log_sum(forward_temp, iter_slog(forward[i][j][state][t]))
                forward_position[i][j][state] = forward_temp
                print "---------------------------", i, j, states[state], forward_position[i][j][state]
    check = 0
    for i in range(len(read)):
        for j in range(len(reference)):
            check = log_sum(check, iter_slog(forward_position[i][j]))
    print check
    return forward


def count_backward(read_tmp, reference_tmp, model):
    read = ' ' + read_tmp
    reference = ' ' + reference_tmp
    len_reference = len(reference_tmp)  # true length
    len_read = len(read_tmp)
    print "Len_read: ", len_read, "len_reference: ", len_reference
    # position and create len_sequence to remember true length
    states = {0: 'Match', 1: 'Deletion', 2: 'Insertion', 3: 'Begin', 4: 'End'}
    max_hp_read = [0]   # because read start from ' '
    max_hp_read.extend(process_sequence_start(read_tmp))    # information about the lengthest HP at each position
    max_hp_ref = [0]   # because reference start from ' '
    max_hp_ref.extend(process_sequence_start(reference_tmp))
    ref_hmm_blocks = hmm_block(reference)     # information about HMM blocks
    backward = float("-inf")*numpy.ones(shape=[len(read), len(reference), 5, max(max_hp_read) + 1, max(max_hp_ref) + 1],
                                        dtype=float)
    print "Shape: ", backward.shape
    backward_position = float("-inf")*numpy.ones(shape=[len(read), len(reference), 5], dtype=float)

    # initialize
    if reference[len_reference] == read[len_read]:
        for i in range(1, length_last_hp(read) + 1):
            for j in range(1, length_last_hp(reference) + 1):
                backward[len_read][len_reference][0][i][j] = 0
                print "-----------read: ", len_read, "len: ", i, "ref: ", len_reference, \
                    "len: ", j, states[0], backward[len_read][len_reference][0][i][j]
    else:
        for i in range(length_last_hp(read) + 1):
            backward[len_read][len_reference][2][i][0] = 0
            print "-----------read: ", len_read, "len: ", i, "ref: ", len_reference, \
                    "len: ", 0, states[2], backward[len_read][len_reference][2][i][0]
        for j in range(length_last_hp(reference) + 1):
            backward[len_read][len_reference][1][0][j] = 0
            print "-----------read: ", len_read, "len: ", 0, "ref: ", len_reference, \
                    "len: ", j, states[1], backward[len_read][len_reference][1][0][j]

    def iterative_sum(prev_state, i, j, d_i_max, d_j_max, read_base, reference_base):
        """
        Count backward variable
        :param prev_state: state of backward variable
        :param i: start of read suffix
        :param j: start of reference suffix
        :param d_i_max: maximum length of HP, starting at position i
        :param d_j_max: maximum length of HP, starting at position j
        :return:
        """
        result = float("-inf")
        for state in [0, 1, 2]:

            transition = eln(model.HMM[ref_hmm_blocks[j]].transition(states[prev_state], states[state]))
            if transition == float("-inf"):
                continue

            for d_i in range(d_i_max + 1):

                hp_output = homopolymer() if d_i == 0 else homopolymer(read_base, d_i)
                for d_j in range(d_j_max + 1):

                    hp_input = homopolymer() if d_j == 0 else homopolymer(reference_base, d_j)
                    emission = eln(model.HMM[ref_hmm_blocks[j]].emission(hp_input, hp_output, states[state]))
                    if emission == float("-inf"):
                        continue
                    #print hp_input.base, hp_output.base, states[state], emission
                    result = log_sum(result,
                                     iter_plog([transition, emission, backward[i + d_i][j + d_j][state][d_i][d_j]]))
        return result

    for i in range(len_read - 1, -1, -1):  # read position

        for j in range(len_reference - 1, -1, -1):     # reference position

            for k in range(max_hp_read[i + 1], -1, -1):

                for l in range(max_hp_ref[j + 1], - 1, -1):

                    if l == 0 and k == 0:
                        continue
                    elif k == 0:    # Deletion case
                        backward[i][j][1][k][l] = iterative_sum(1, i, j, max_hp_ref[i], max_hp_ref[j + 1],
                                                                read[i], reference[j + 1])
                        print "-----------read: ", i, read[i], "k: ", k, "ref: ", j, read[j], "l: ", l, states[1], backward[i][j][1][k][l]
                    elif l == 0:    # Insertion case
                        backward[i][j][2][k][l] = iterative_sum(2, i, j, max_hp_ref[i + 1], max_hp_ref[j],
                                                                read[i + 1], reference[j])
                        print "-----------read: ", i, read[i], "k: ", k, "ref: ", j, read[j], "l: ", l, states[2], backward[i][j][2][k][l]
                    else:
                        if read[i] == reference[j]:     # if hp ending at i and hp ending at j have same base
                            backward[i][j][0][k][l] = iterative_sum(0, i, j, max_hp_ref[i + 1], max_hp_ref[j + 1],
                                                                read[i + 1], reference[j + 1])
                            print "-----------read: ", i, read[i], "k: ", k, "ref: ", j, reference[j], "l: ", l, states[0], backward[i][j][0][k][l]
                        print "-----------read: ", i, read[i], "k: ", k, "ref: ", j, reference[j], "l: ", l, states[0], backward[i][j][0][k][l]

    check = 0
    for i in range(len(read)):
        for j in range(len(reference)):
            check = log_sum(check, iter_slog(backward_position[i][j]))
    print check
    return backward




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

    #count_forward(read, reference, hmm_test)
    count_backward(read, reference, hmm_test)

    return 0

main()