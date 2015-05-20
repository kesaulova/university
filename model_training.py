import EM_algorithm as em
import re
import hmm
import Viterbi as vit
import numpy
from inspect import currentframe, getframeinfo
import time
import multiprocessing as mp

def form_train_parameters(fasta_name):
    """
    Extract first len_train_set records from sam-file and form train_set. Also count distribulion of hp length across
    reference and scale paramter for laplace distribution.
    :param len_train_set: desirable lenth of training set
    :return: training set, observed length distribution, scale parameter and maximum length of HP in reference and read
    """

    def count_occurrences(ref, max_len):
        """
        Count frequency of HP all length along with reference, independent of HP base
        :param ref: reference string
        :param max_len: length of max HP from reference
        :return:
        """
        length_distrib = numpy.zeros(shape=[max_len + 1], dtype=float)
        hp_length = 1
        current_base = ref[0]
        for i in xrange(1, len(ref)):
            if ref[i] == current_base:
                hp_length += 1
            else:
                length_distrib[hp_length] += 1
                current_base = ref[i]
                hp_length = 1
        length_distrib = length_distrib / sum(length_distrib)
        return length_distrib

    def test_count_b(max_len):
        """
        Take parameters from article and count b. b = c_0 + c_1*l^c_2
        :param max_length: maximum length of homopolymer
        :return: array of length max_length + 1 - for convenience call b[l]
        """
        c_0 = 0.0665997
        c_1 = 0.0471694
        c_2 = 1.23072
        res = [0]
        for l in xrange(max_len):
            res += [c_0 + c_1 * l**c_2]
        return res

    def hp_max_len(sequence):
        """
        Detect maximum length of HP along the given sequence.
        :param sequence: string
        :return: length of lengthest HP in sequence
        """
        curr_base = sequence[0]
        max_count = 0
        curr_count = 1
        for i in xrange(1, len(sequence)):
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
                quality_list.extend([ord(quality_string[k]) - 33 for k in xrange(i, i + l)])
                i += l
                g += l
            elif t == 'D':  # skip 'l' number of coordinates in reference
                # quality_list.extend([0]*l)
                reference_edit += reference[g:g + l]
                g += l
            elif t == 'I':  # insertion of 'l' length
                # insertions.append((i, i + l))
                read_edit += read[i:i + l]
                quality_list.extend([ord(quality_string[k]) for k in xrange(i, i + l)])
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

    def count_occurrences_bases(ref, max_len):
        """
        Count frequency of HP all length along with reference, depending on HP base
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

    fasta_string = '0'
    for line in open(fasta_name, 'r'):
        if line[0] != '>':
            fasta_string += line.rstrip('\n')
    max_hp_ref = hp_max_len(fasta_string)
    observed_freq = count_occurrences(fasta_string[1:], max_hp_ref)
    assert (max_hp_ref <= 12), "Lengthest HP in reference longer than 12! It is " + str(max_hp_ref)

    b_scale = test_count_b(max_hp_ref)

    # training_set = []
    # fasta_length = len(fasta_string)
    #
    # tr_set = open("training_set.txt", 'w')
    # i = 0   # counter for number of item in training set
    #
    # for read in open('B22-730.sam'):
    #     if i > len_train_set:
    #         break
    #     if read[0] == "@":
    #         continue
    #
    #     sam_fields = re.split('\t', read.rstrip('\n'))
    #     read_seq = sam_fields[9]
    #
    #     # Throw away reads, contains HP with length more than maximum hp length in reference
    #     if hp_max_len(read_seq) > max_hp_ref:
    #         continue
    #     gen_read, path = hmm.create_sequence(model, 120, read_seq[:min(100, len(read_seq))])
    #     if hp_max_len(gen_read) > max_hp_ref:
    #         continue
    #     # generated_set.append([gen_read, read_seq])
    #
    #     pos = int(sam_fields[3])
    #     if pos + len(read_seq) + 150 < fasta_length:
    #         read, reference, read_quality, end = split_cigar(sam_fields[5], read_seq,
    #                                         fasta_string[pos:pos + len(read_seq) + 150], sam_fields[10])
    #         if read_quality != [] and len(read) == len(sam_fields[10]):
    #             write_set(read, fasta_string[pos:pos + end], sam_fields[10], tr_set)
    #             training_set.append([read, reference, sam_fields[10]])
    #             i += 1
    #     else:
    #         continue
    # tr_set.close()
    return observed_freq[:], b_scale, max_hp_ref
    # return generated_set, observed_freq['A'][:], b_scale, max_hp_ref


def region_train_set(begin, min_len):
    """
    Detect reads, relating to desired region in reference string
    :param begin: start region
    :param min_len: begin + min_len - end of region
    :return: train set
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
                quality_list.extend([ord(quality_string[k]) - 33 for k in xrange(i, i + l)])
                i += l
                g += l
            elif t == 'D':  # skip 'l' number of coordinates in reference
                # quality_list.extend([0]*l)
                reference_edit += reference[g:g + l]
                g += l
            elif t == 'I':  # insertion of 'l' length
                # insertions.append((i, i + l))
                read_edit += read[i:i + l]
                quality_list.extend([ord(quality_string[k]) for k in xrange(i, i + l)])
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

    def write_set(read, reference, file):
        """
        Write training set to file
        :param read: read string
        :param reference: reference string
        :param quality: quality array
        :param file: file to write
        :return:
        """
        tmp = read + '\t' + reference + '\n'
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
        for i in xrange(1, len(sequence)):
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

    # process fasta-file
    fasta_string = '0'
    for line in open("DH10B-K12.fasta", 'r'):
        if line[0] != '>':
            fasta_string += line.rstrip('\n')
    max_hp_ref = hp_max_len(fasta_string)
    assert (max_hp_ref <= 12), "Lengthest HP in reference longer than 12! It is " + str(max_hp_ref)

    training_set = []
    tr_set = open("C11_training_set.txt", 'a')
    i = 0
    for read in open('C11.sam'):
        i += 1
        if read[0] == "@":
            continue

        sam_fields = re.split('\t', read.rstrip('\n'))
        pos = int(sam_fields[3])
        read_seq = sam_fields[9]
        if (pos < begin) and (pos + len(read_seq)) > (begin + min_len):
            # Throw away reads, contains HP with length more than maximum hp length in reference
            if hp_max_len(read_seq) > max_hp_ref:
                continue

            # Read without quality
            read, reference, read_quality, end = split_cigar(sam_fields[5], read_seq,
                                            fasta_string[pos:pos + len(read_seq) + 150], sam_fields[10])
            # if read_quality != [] and len(read) == len(sam_fields[10]):
            # print "Difference: ", len(read) - len(fasta_string[pos:pos + end]), len(read), end
            if read == fasta_string[pos:pos + end]:
                print "They are equal"
            else:
                write_set(read, fasta_string[pos:pos + end], tr_set)
                training_set.append([read, reference])
            # else:
            #     print "Here else"
            #     print len(sam_fields[10]), len(read), sam_fields[5]

    tr_set.close()
    print i, len(training_set)
    return training_set,  max_hp_ref


def count_lik_outer(pair):
    """
    Additional function. Run Viterbi for read, reference.
    :param pair: list [read, reference, desired_length_of_sequence, HMM model]
    :return: probability of most probable hidden path for given read and reference
    """
    model = pair[3]
    cur_len = pair[2]
    path, probability = vit.viterbi_path(pair[0][:cur_len], pair[1][:cur_len], model)
    return probability


def count_likelihood(train_set, model, cur_len, num_proc):
    """
    Count likelihood of training set (set of pairs). For each pair run Viterbi and get probability of most probable path
    in given model.
    :param train_set: set of pairs [read, reference]
    :param model: HMM nodel
    :param cur_len: desired length of string - when we don't want to process whole string
    :param num_proc: number of processes that can be used
    :return: number - likelihood for set
    """
    train_set_extended = []
    for item in train_set:
        new_item = item + [cur_len] + [model]
        train_set_extended.append(new_item)
    pool = mp.Pool(processes=num_proc)
    probs = pool.map(count_lik_outer, train_set_extended)
    return sum(probs)


def generate_set(model, ref, set_size, len_seq, file_name):
    """
    Create set from reference string, based on parameters of given HMM model.
    :param model: HMM model
    :param ref: reference string
    :param set_size: size of desired set
    :param len_seq: with this parameter can limit length of reference string
    :param file_name: write set to this file
    :return: training set
    """
    f_set = open(file_name, 'w')
    result_set = []
    for i in range(set_size):
        gen_read, path = hmm.create_sequence(model, len_seq + 20, ref[:min(len_seq, len(ref))])
        result_set.append([gen_read, ref])
        f_set.set.writelines([gen_read, '\t', ref])
    return result_set


def read_set_from_file(file_name):
    """
    Read training et from file. File must have records 'read \t reference'
    :param file_name:
    :return:
    """
    result = []
    for lines in open(file_name):
        rec = re.split('\t', lines.rstrip('\n'))
        result.append([rec[0], rec[1]])
    return result


def write_likelihood(f_name, val):
    """
    Add record with given value in file.
    :param f_name: file name
    :param val: current likelihood
    :return: nothing
    """
    f = open(f_name, 'a')
    f.writelines(["\nModel likelihood: ", str(round(val, 4))])
    f.close()
    return 0


def main():
    """
    Implement training of HMM.
    :param sample_data: {(r^d, t^d), d = 1..N} - N pair of reads and mapped reference segments
    :param model: start model
    :return:
    """
    sequence = "GCTGCATGATATTGAAAAAATATCACCAAATAAAAAACGCCTTAGTAAGTATTTTTCAGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGA" \
               "TTAAAAAAGAGTGTCTGATAGCAGCTTCTGAACTGGTTACCTGCCGTGAGTAAATTAAAATTTTATTGACTTAGGTCACTAAATACTTTAACCAATATAGG" \
               "CATAGCGCACAGACAGATAAAAATTACAGAGTACACAACATCCATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTG" \
               "CGGGCTGACGACGTACAGGAAACACAGAAAAAAGCCCGCTAC"
    cur_len = 100500
    sigma = 1
    max_iterations = 80
    lik = []    # list of likelihood for models
    num_proc = 2
    fasta_name = "DH10B-K12.fasta"
    gen_set_filename = "generated_train_set.txt"
    train_set_filename = "C11_training_set.txt"
    par_file_name = "HMM_parameters.txt"
    is_generate_set = False

    # clean file for parameters
    parameters_file = open(par_file_name, 'w')

    hmm_model = hmm.HmmModel(par_file_name)
    print "Test model created"

    # freq, b, max_hp_ref = form_train_set(fasta_name)
    freq, cheat_b, max_hp_ref = form_train_parameters(fasta_name)

    if is_generate_set:
        generate_set(hmm_model, sequence, max_iterations, len(sequence), gen_set_filename)
        train_set = read_set_from_file(gen_set_filename)
        print "Generated train set formed, size: ", max_iterations, ", sequence length: ", cur_len
        parameters_file.writelines(["Generated train set formed, size: ", str(max_iterations), ", sequence length: ",
                                    str(cur_len), '\n'])
    else:
        train_set = read_set_from_file(train_set_filename)
        print "Train set formed from file ", train_set_filename, ", set size: ", max_iterations, \
            ", sequence length: ", cur_len
        parameters_file.writelines(["Train set formed, size: ", str(max_iterations), ", sequence length: ",
                                    str(cur_len), '\n'])

    print "Max length from reference: ", max_hp_ref
    parameters_file.writelines(["Max length from reference: ", str(max_hp_ref), '\n'])
    parameters_file.close()

    lik.append(count_likelihood(train_set, hmm_model, cur_len, num_proc))
    write_likelihood(par_file_name, lik[0])
    print "Likelihood: ", lik[0]

    i = 1
    while True:
        print "Step: ", i
        parameters_file.writelines(["Step: ", str(i), '\n'])
        ins_base, len_match, len_ins, trans, b, sigma = em.update_parameters(train_set, hmm_model,
                                                max_hp_ref, cheat_b, sigma, freq, cur_len, max_iterations, i, num_proc)

        # ins_base, len_match, len_ins, trans, b, sigma = em.update_parameters(train_set, hmm_model, max_hp_ref, b,
        #                                                                      sigma, freq, cur_len, max_iterations, i)

        trans[3, ] = em.get_eln([0.9, 0.05, 0.05, 0, 0.0001])
        trans[4, ] = em.get_eln([0.0, 0.0, 0.0, 0.0, 1.0])
        hmm_model = hmm.HmmModel(par_file_name, ins_base, len_match, len_ins, trans)
        lik.append(count_likelihood(train_set, hmm_model, cur_len, num_proc))
        write_likelihood(par_file_name, lik[i])
        print "Model created, ", "Likelihood: ", lik[i]
        print lik
        if abs(lik[i] - lik[i]) < 0.05 or i > 15:
            break
        i += 1

    return 0

main()