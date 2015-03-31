import addmath
import hmm
import Viterbi

discrete_distribution = addmath.discrete_distribution
homopolymer = hmm.homopolymer
HmmModel = hmm.HmmModel
create_sequence = hmm.create_sequence
alphabet = {'A', 'C', 'G', 'T'}
extendedAlphabet = {'A', 'C', 'G', 'T', '-'}



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



