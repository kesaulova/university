from scipy.integrate import quad
from scipy.stats import laplace, lognorm
import pylab
import numpy


def count_length_call_match(max_hp_length, scale, freq):
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
            num = laplace.pdf(x, loc=l, scale=scale[l]) * laplace.pdf(x, loc=k, scale=scale[k]) * freq[k]
            denom = sum([freq[i] * laplace.pdf(x, loc=i, scale=scale[i]) for i in range(1, max_hp_length + 1)])
            return num/denom

        def normalize(item, max_len):
            """
            Normalize length call matrix (sum values in one row must be 1)
            :param item: square matrix
            :param max_len: number of rows/columns
            :return:
            """
            for i in range(max_len):
                item[i, ] = item[i, ] / sum(item[i, ])
            return item

        for l in range(1, max_hp_length + 1):
            for k in range(1, max_hp_length + 1):
                result[l - 1, k - 1] = quad(lcall, 0, max_hp_length, args=(l, k))[0]
        result = normalize(result, max_hp_length)
        return result

freq = [0.0, 0.7152478454907308, 0.19311689366656737, 0.06231522667579097, 0.019076114917007392, 0.0071816975637614125, 0.0023397441420240695, 0.0005710895749075056, 0.00014154159722923093, 9.846371981163891e-06, 0.0, 0.0, 0.0, 0.0, 0.0]
bad_b = [0, 0.44497037740895828, 0.46158480525291834, 0.5240247782375812, 0.60184556639144848, 0.7130486599148651, 0.778072471554066, float('NaN'), float('NaN'), 1.31400676193967, float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN')]
b = [0, 0.44497037740895828, 0.46158480525291834, 0.5240247782375812, 0.60184556639144848, 0.7130486599148651, 0.778072471554066, 0.81058437737366651, 0.82684033028346682, 1.31400676193967, 1.5575899777677717, 1.6793815856818226, 1.7402773896388481, 1.770725291617361, 1.7859492426066175]
sigma = 4
new_sigma = 0.8

def lcall(x, l, k):
    """
    Integrated function
    :param x: f, flow intensity
    :param l: length of input hp
    :param k: length of output hp
    :return: counted function
    """
    num = laplace.pdf(x, loc=l, scale=b[l]) * laplace.pdf(x, loc=k, scale=b[k]) * freq[k]
    denom = sum([freq[i] * laplace.pdf(x, loc=i, scale=b[i]) for i in range(1, 13 + 1)])
    return num/denom


def p_k_f(x, k):
    return laplace.pdf(x, loc=k, scale=b[k]) * freq[k] / sum([freq[i] * laplace.pdf(x, loc=i, scale=b[i]) for i in range(1, 14)])


def p_f_l(x, l):
    return laplace.pdf(x, loc=l, scale=b[l])


def lcall_ins(x, k):
    num = lognorm.pdf(x, 1, loc=0, scale=sigma) * laplace.pdf(x, loc=k, scale=b[k]) * freq[k]
    denom = sum([freq[i] * laplace.pdf(x, loc=i, scale=b[i]) for i in range(1, 14)])
    return num/denom

xx = numpy.arange(0, 13, 0.01)
for l in range(1, 13):
    # plot p(k | f)
    fig = pylab.figure()
    yy = [p_k_f(x, l) for x in xx]
    pylab.plot(xx, yy)
    st = "p(k | f), k = " + str(l)
    pylab.title(st)
    pylab.xlabel("Flow intensity")
    pylab.ylabel("Probability")
    nm = "/Users/kesaulova/Documents/PyCharm/Diploma/p_k_f/" + str(l) + ".png"
    pylab.savefig(nm)
    pylab.close(fig)

    # plot p(f | l)
    fig = pylab.figure()
    yy = [p_f_l(x, l) for x in xx]
    pylab.plot(xx, yy)
    st = "p(f | l), l = " + str(l)
    pylab.title(st)
    pylab.xlabel("Flow intensity")
    pylab.ylabel("Probability")
    nm = "/Users/kesaulova/Documents/PyCharm/Diploma/p_f_l/" + str(l) + ".png"
    pylab.savefig(nm)
    pylab.close(fig)

    # plot p(k | l)
    fig = pylab.figure()
    yy = [numpy.mean([lcall(x, l, k) for x in xx]) for k in range(1, 14)]
    pylab.plot(range(1, 14), yy)
    st = "p(k | l),  l = " + str(l)
    pylab.title(st)
    pylab.xlabel("Length of output HP")
    pylab.ylabel("Probability")
    nm = "/Users/kesaulova/Documents/PyCharm/Diploma/p_k_l/" + str(l) + ".png"
    pylab.savefig(nm)
    pylab.close(fig)

xx = numpy.arange(0.0001, 13, 0.01)
# p(k | 0)
fig = pylab.figure()
yy = [numpy.mean([lcall_ins(x, k) for x in xx]) for k in range(1, 14)]
pylab.plot(range(1, 14), yy)
st = "p(k | 0)"
pylab.title(st)
pylab.xlabel("Length of output HP")
pylab.ylabel("Probability")
nm = "/Users/kesaulova/Documents/PyCharm/Diploma/p_k_l/0.png"
pylab.savefig(nm)
pylab.close(fig)