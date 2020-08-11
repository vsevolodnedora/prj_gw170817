#
#
#
import numpy as np
from scidata import units

constant_length = 1.47671618189 # geo -> rm
constant_rho = 6.176269145886162e+17#1.61910042516e-18 # geo->cgs
constant_time = 0.004925794970773136  # to to to ms
constant_energy = 1787.5521500932314
constant_volume = 2048

def standard_div(x_arr):
    x_arr = np.array(x_arr, dtype=float)
    n = 1. * len(x_arr)
    mean = sum(x_arr) / n
    tmp = (1 / (n-1)) * np.sum((x_arr - mean) ** 2)
    return mean, np.sqrt(tmp)

def x_y_z_sort(x_arr, y_arr, z_arr=np.empty(0, ), sort_by_012=0):
    '''
    RETURNS x_arr, y_arr, (z_arr) sorted as a matrix by a row, given 'sort_by_012'
    :param x_arr:
    :param y_arr:
    :param z_arr:
    :param sort_by_012:
    :return:
    '''

    # ind = np.lexsort((x_arr, y_arr))
    # tmp = [(x_arr[i],y_arr[i]) for i in ind]
    #
    # print(tmp); exit(1)

    if len(z_arr) == 0 and sort_by_012 < 2:
        if len(x_arr) != len(y_arr):
            raise ValueError('len(x)[{}]!= len(y)[{}]'.format(len(x_arr), len(y_arr)))

        x_y_arr = []
        for i in range(len(x_arr)):
            x_y_arr = np.append(x_y_arr, [x_arr[i], y_arr[i]])
        # print(x_y_arr.shape)

        x_y_sort = np.sort(x_y_arr.view('float64, float64'), order=['f{}'.format(sort_by_012)], axis=0).view(
            np.float)
        x_y_arr_shaped = np.reshape(x_y_sort, (int(len(x_y_sort) / 2), 2))
        # print(x_y_arr_shaped.shape)
        _x_arr = x_y_arr_shaped[:, 0]
        _y_arr = x_y_arr_shaped[:, 1]
        assert len(_x_arr) == len(x_arr)
        assert len(_y_arr) == len(y_arr)

        return _x_arr, _y_arr

    if len(z_arr) > 0 and len(z_arr) == len(y_arr):
        if len(x_arr) != len(y_arr) or len(x_arr) != len(z_arr):
            raise ValueError('len(x)[{}]!= len(y)[{}]!=len(z_arr)[{}]'.format(len(x_arr), len(y_arr), len(z_arr)))

        x_y_z_arr = []
        for i in range(len(x_arr)):
            x_y_z_arr = np.append(x_y_z_arr, [x_arr[i], y_arr[i], z_arr[i]])

        x_y_z_sort = np.sort(x_y_z_arr.view('float64, float64, float64'), order=['f{}'.format(sort_by_012)],
                             axis=0).view(np.float)
        x_y_z_arr_shaped = np.reshape(x_y_z_sort, (int(len(x_y_z_sort) / 3), 3))
        return x_y_z_arr_shaped[:, 0], x_y_z_arr_shaped[:, 1], x_y_z_arr_shaped[:, 2]

def find_nearest_index(array, value):
    ''' Finds index of the value in the array that is the closest to the provided one '''
    idx = (np.abs(array - value)).argmin()
    return idx

def fit_polynomial(x, y, order, depth, new_x=np.empty(0, ), print_formula=True):
    '''
    RETURNS new_x, f(new_x)
    :param x:
    :param y:
    :param order: 1-4 are supported
    :return:
    '''

    x = np.array(x)
    y = np.array(y)

    f = None
    lbl = None

    if not new_x.any():
        new_x = np.mgrid[x.min():x.max():depth * 1j]

    if order == 1:
        fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
        f = np.poly1d(fit)
        lbl = '({}) + ({}*x)'.format(
            "%.4f" % f.coefficients[1],
            "%.4f" % f.coefficients[0]
        )
        # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
        # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
    if order == 2:
        fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
        f = np.poly1d(fit)
        lbl = '({}) + ({}*x) + ({}*x**2)'.format(
            "%.4f" % f.coefficients[2],
            "%.4f" % f.coefficients[1],
            "%.4f" % f.coefficients[0]
        )
        # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
        # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
    if order == 3:
        fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
        f = np.poly1d(fit)
        lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3)'.format(
            "%.4f" % f.coefficients[3],
            "%.4f" % f.coefficients[2],
            "%.4f" % f.coefficients[1],
            "%.4f" % f.coefficients[0]
        )
        # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
        # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
    if order == 4:
        fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
        f = np.poly1d(fit)
        lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3) + ({}*x**4)'.format(
            "%.4f" % f.coefficients[4],
            "%.4f" % f.coefficients[3],
            "%.4f" % f.coefficients[2],
            "%.4f" % f.coefficients[1],
            "%.4f" % f.coefficients[0]
        )
        # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
        # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
    if order == 5:
        fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
        f = np.poly1d(fit)
        lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3) + ({}*x**4) + ({}*x**5)'.format(
            "%.4f" % f.coefficients[5],
            "%.4f" % f.coefficients[4],
            "%.4f" % f.coefficients[3],
            "%.4f" % f.coefficients[2],
            "%.4f" % f.coefficients[1],
            "%.4f" % f.coefficients[0]
        )
        # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
        # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')
    if order == 6:
        fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
        f = np.poly1d(fit)
        lbl = '({}) + ({}*x) + ({}*x**2) + ({}*x**3) + ({}*x**4) + ({}*x**5) + ({}*x**6)'.format(
            "%.4f" % f.coefficients[6],
            "%.4f" % f.coefficients[5],
            "%.4f" % f.coefficients[4],
            "%.4f" % f.coefficients[3],
            "%.4f" % f.coefficients[2],
            "%.4f" % f.coefficients[1],
            "%.4f" % f.coefficients[0]
        )
        # fit_x_coord = np.mgrid[(x.min()):(x.max()):depth*1j]
        # plt.plot(fit_x_coord, f(fit_x_coord), '--', color='black')

    if not order in [1, 2, 3, 4, 5, 6]:
        fit = np.polyfit(x, y, order)  # fit = set of coeddicients (highest first)
        f = np.poly1d(fit)
        # raise ValueError('Supported orders: 1,2,3,4 only')

    if print_formula:
        print(lbl)

    return new_x, f(new_x)

if __name__ == "__main__":

    from scidata import utils
    from scidata import units
