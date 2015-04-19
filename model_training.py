import EM_algorithm as em
import re
import hmm
import Viterbi as vit
from inspect import currentframe, getframeinfo


def form_train_set(len_train_set):
    """
    Extract first len_train_set records from sam-file and form train_set. Also count distribulion of hp length across
    reference and scale paramter for laplace distribution.
    :param len_train_set: desirable lenth of training set
    :return: training set, observed length distribution, scale parameter and maximum length of HP in reference and read
    """

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
                # quality_list.extend([0]*l)
                reference_edit += reference[g:g + l]
                g += l
            elif t == 'I':  # insertion of 'l' length
                # insertions.append((i, i + l))
                read_edit += read[i:i + l]
                quality_list.extend([ord(quality_string[k]) for k in range(i, i + l)])
                i += l
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
    
    def hp_max_len(sequence):
        """
        Detect maximum length of HP along the given sequence.
        :param sequence: string
        :return: length of lengthest HP in sequence
        """
        curr_base = sequence[0]
        max_count = 0
        curr_count = 1
        for i in range(1, len(sequence)):
            if sequence[i] == curr_base:
                curr_count += 1
            else:
                if curr_count > max_count:
                    max_count = curr_count
                curr_base = sequence[i]
                curr_count = 1
        if curr_count > max_count:
            max_count = curr_count
        return max_count

    #
    # def count_lengthest(sequence, a=15):
    #     count = 0
    #     curr_count = 1
    #     curr_base = sequence[0]
    #     for i in range(1, len(sequence)):
    #         if sequence[i] == curr_base:
    #             curr_count += 1
    #         else:
    #             if curr_count == a:
    #                 count += 1
    #             curr_base = sequence[i]
    #             curr_count = 1
    #     return count
    #
    # temporary!!!

    def count_occurrences(ref, max_len):
        """
        Count frequency of HP all length along with reference
        :param ref: reference string
        :param max_len: length of max HP from reference
        :return:
        """
        length_distrib = {'A': [0] * (max_len + 1), 'C': [0] * (max_len + 1),
                          'G': [0] * (max_len + 1), 'T': [0] * (max_len + 1)}
        hp_length = 1
        current_base = ref[0]
        for i in xrange(1, len(ref)):
            if ref[i] == current_base:
                hp_length += 1
            else:
                length_distrib[current_base][hp_length] += 1
                current_base = ref[i]
                hp_length = 1
        for key in length_distrib:
            length_distrib[key] = [item/float(sum(length_distrib[key])) for item in length_distrib[key]]
        return length_distrib

    # process fasta-file
    fasta_string = '0'
    for line in open("DH10B-K12.fasta", 'r'):
        if line[0] != '>':
            fasta_string += line.rstrip('\n')
    max_hp_ref = hp_max_len(fasta_string)
    assert (max_hp_ref <= 12), "Lengthest HP in reference longer than 12! It is " + str(max_hp_ref)

    training_set = []
    fasta_length = len(fasta_string)
    observed_freq = count_occurrences(fasta_string[1:], max_hp_ref)
    tr_set = open("training_set.txt", 'w')
    i = 0   # counter for number of item in training set

    for read in open('B22-730.sam'):
        if i > len_train_set:
            break
        if read[0] == "@":
            continue

        sam_fields = re.split('\t', read.rstrip('\n'))
        read_seq = sam_fields[9]

        # Throw away reads, contains HP with length more than maximum hp length in reference
        if hp_max_len(read_seq) > max_hp_ref:
            continue

        pos = int(sam_fields[3])
        if pos + len(read_seq) + 150 < fasta_length:
            read, reference, read_quality, end = split_cigar(sam_fields[5], read_seq,
                                            fasta_string[pos:pos + len(read_seq) + 150], sam_fields[10])
            if read_quality != [] and len(read) == len(sam_fields[10]):
                write_set(read, fasta_string[pos:pos + end], sam_fields[10], tr_set)
                training_set.append([read, reference, sam_fields[10]])
                i += 1
                # temp = [count_lengthest(read_seq, i) for i in range(16)]
                # count_lengthest_read = [0] + [count_lengthest_read[i] + temp[i] for i in range(1, 16)]
        else:
            continue
    tr_set.close()
    b_scale = test_count_b(max_hp_ref)
    return training_set, observed_freq['A'][:], b_scale, max_hp_ref


def count_likelihood(train_set, model):
    result = 0
    for pair in train_set:
        path, probability = vit.viterbi_path(pair[0][:30], pair[1][:30], model)
        result += probability
    return result


def training(train_set):
    likelihood = []
    model = hmm.HmmModel()  # initial
    train_set, freq, b, max_ref = form_train_set()
    sigma = 4
    print "train_set formed", getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno

    while True:
        # base_call, len_call_match, len_call_ins, trans_prob
        ins_base, len_match, len_ins, trans = em.update_parameters(train_set, model, max_ref, b, sigma, freq)
        new_model = hmm.HmmModel(ins_base, len_match, len_ins, trans)
        curr_likelihood = count_likelihood(train_set, new_model)
        print likelihood[len(likelihood) - 1], curr_likelihood
        likelihood.append(curr_likelihood)

            # new_model = hmm.HmmModel(ins_base_call, length_call_match, length_call_ins, transition_matrix)


def main():
    """
    Implement training of HMM.
    :param sample_data: {(r^d, t^d), d = 1..N} - N pair of reads and mapped reference segments
    :param model: start model
    :return:
    """
    hmm_test = hmm.HmmModel()

    train_set, freq, b, max_hp_ref = form_train_set(100)
    print freq
    print "train_set formed"
    print "Max length from reference: ", max_hp_ref

    print count_likelihood(train_set, hmm_test)

    ins_base, len_match, len_ins, trans = em.update_parameters(train_set, hmm_test, max_hp_ref, b, 4, freq)
    trans[3, ] = em.get_eln([0.2999, 0.3, 0.4, 0, 0.0001])
    trans[4, ] = em.get_eln([0.0, 0.0, 0.0, 0.0, 1.0])
    new_model = hmm.HmmModel(ins_base, len_match, len_ins, trans)
    print "Model created"
    print count_likelihood(train_set, new_model)
    # it returns ins_base_call, length_call_match, length_call_ins, transition_matrix

    return 0

main()