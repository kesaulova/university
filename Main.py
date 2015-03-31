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
    ref = [homopolymer('A', 3)]
    bases = ['A', 'C', 'G', 'T']
    for i in range(length):
        base = bases[discrete_distribution([0.25, 0.25, 0.25, 0.25])]
        while base == ref[len(ref) - 1].base:
            base = bases[discrete_distribution([0.25, 0.25, 0.25, 0.25])]
        length = discrete_distribution([10., 5., 1., 0.5, 10**(-2), 10**(-3), 10**(-4), 10**(-5), 10**(-6), 10**(-7), 10**(-8), 10**(-9), 10**(-10), 10**(-11)]) + 1
        ref.append(homopolymer(base, length))
    return ref


def test_viterbi_initialize(read, reference, model):
    viterbi_probability, viterbi_backtracking = Viterbi.viterbi_path(hmm.homopolymer_to_nucleotide(read), reference, model)
    print viterbi_probability[0][2][0][1]
    print viterbi_probability[0][1][0][1]
    print viterbi_probability[0][0][0][1]

    print viterbi_backtracking[0][2][0][1]
    print viterbi_backtracking[0][1][0][1]
    print viterbi_backtracking[0][0][0][1]

    print viterbi_probability[0][0][1][2]
    print viterbi_probability[1][0][1][2]
    print viterbi_probability[2][0][1][2]

    print viterbi_backtracking[0][0][1][2]
    print viterbi_backtracking[1][0][1][2]
    print viterbi_backtracking[2][0][1][2]

    print viterbi_probability[0][0][1][0]
    print viterbi_probability[1][0][1][1]
    print viterbi_probability[0][1][0][0]
    print viterbi_probability[0][1][0][2]
    return 0


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

    test_viterbi_initialize(sequence, reference_test, hmm_test)
    viterbi_probability, viterbi_backtracking = Viterbi.viterbi_path(hmm.homopolymer_to_nucleotide(sequence), reference_test, hmm_test)
    print Viterbi.parse_viterbi(viterbi_probability, viterbi_backtracking, len(reference_test), hmm.homopolymer_to_nucleotide(sequence))

    return 0


main()



