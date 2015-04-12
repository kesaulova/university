import addmath
import hmm
import Viterbi
import re
import EM_algorithm as em

discrete_distribution = addmath.discrete_distribution
homopolymer = hmm.homopolymer
HmmModel = hmm.HmmModel
create_sequence = hmm.create_sequence
alphabet = {'A', 'C', 'G', 'T'}
extendedAlphabet = {'A', 'C', 'G', 'T', '-'}



def form_dataset():
    training_set = []

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

    # open fasta-file

    fasta_string = '0'
    for line in open("DH10B-K12.fasta", 'r'):
        if line[0] != '>':
            fasta_string += line.rstrip('\n')
    fasta_length = len(fasta_string)

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
            print len(read), len(read_quality), len(read_seq), len(samRecordFields[10])
            if read_quality != [] and len(read) == len(samRecordFields[10]):
                write_set(read, fasta_string[pos:pos + end], samRecordFields[10], set)
                training_set.append([read, reference, samRecordFields[10]])

        else:
            continue
    tr_set.close()
    return training_set



def test_reference(length):
    """
    Create sequence (with givenlength) of homopolymers
    :param length: desired length of sequence
    :return:
    """
    ref = [homopolymer('A', 3)]
    bases = ['A', 'C', 'G', 'T']
    for i in range(length):
        base = bases[discrete_distribution([0.25, 0.25, 0.25, 0.25])]
        while base == ref[len(ref) - 1].base:
            base = bases[discrete_distribution([0.25, 0.25, 0.25, 0.25])]
        length = discrete_distribution([10., 5., 1., 0.5, 10**(-2), 10**(-3), 10**(-4), 10**(-5), 10**(-6), 10**(-7), 10**(-8), 10**(-9), 10**(-10), 10**(-11)]) + 1
        ref.append(homopolymer(base, length))
    return ref


def main():
    hmm_test = HmmModel()
    reference_test = test_reference(2)
    state, sequence = create_sequence(hmm_test, 20, reference_test)
    print 'State sequence: ', state
    print 'Reference:',
    for i in range(len(reference_test)):
        print '[ ', reference_test[i].base, reference_test[i].length,  ' ] ',
    print '\n Read:',
    for i in range(len(sequence)):
        print '[ ', sequence[i].base, sequence[i].length, ' ] ',
    print '\n Reference:',
    print hmm.homopolymer_to_nucleotide(reference_test)
    print '\n Read:',
    print hmm.homopolymer_to_nucleotide(sequence)

    viterbi_probability, viterbi_backtracking = Viterbi.viterbi_path(hmm.homopolymer_to_nucleotide(sequence),
                                                                    reference_test, hmm_test)
    print Viterbi.parse_viterbi(viterbi_probability, viterbi_backtracking, len(reference_test),
                               hmm.homopolymer_to_nucleotide(sequence))

    return 0




main()



