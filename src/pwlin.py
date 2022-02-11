import math
import numpy as np
from fxpmath import Fxp
from .utils import float2fix_np as float2fix


def apply_pwlin(lut, x, inference_obj, quant_in, quant_out):
    """
    Applies a piecewise linear approximation using a lookup table
    - lut: lookup table of piecewise linear segments (x, y, slope)
    - x: the x value of the point to approximate
    - inference_obj: the NumpyInference Object (needed to call quantization function)
    - quant_in: the input quantization
    - quant_out: the output quantization
    """
    lut_array = np.array(lut)
    lut_indices = np.zeros(x.shape, dtype='int')

    # Find LUT index
    for i in range(len(lut)):
        start_x = lut[i][0]
        lut_indices[x >= start_x] = i
    lut_rows = lut_array[lut_indices, :]
    lut_rows_next = lut_array[lut_indices + 1, :]

    # x_i = inference_obj.quantize(lut_rows[:, 0], Fxp(None, True, 16, quant_in))
    # v_x_i = inference_obj.quantize(lut_rows[:, 1], Fxp(None, True, 16, quant_out))
    # v_x_i_1 = inference_obj.quantize(lut_rows_next[:, 1], Fxp(None, True, 16, quant_out))
    # product_val = inference_obj.quantize(lut_rows[:, 2], Fxp(None, True, 16, 11))
    #
    # delta = inference_obj.quantize(inference_obj.quantize(x - x_i, Fxp(None, True, 16, quant_in)) * product_val,
    #                                Fxp(None, True, 16, 15))




    # float2fix(tmp_val, mantissa_bits)
    v_x_i_1 = float2fix(lut_rows_next[..., 1], quant_out)
    v_x_i = float2fix(lut_rows[..., 1], quant_out)
    x_i = float2fix(lut_rows[..., 0], quant_in)
    product_val = float2fix(lut_rows[..., 2], 11)  # this is final
    delta = float2fix(float2fix((x - x_i), quant_in) * product_val, 15)



    ret_val =  v_x_i_1 * delta + v_x_i * (1 - delta)
    # saturation outside of [-3,3]
    index_minus_3 = np.where(x < -3)
    ret_val[index_minus_3] = 0
    index_3 = np.where(x > 3)
    ret_val[index_3] = x[index_3]
    return ret_val



# ---------------- LUT Function Approximations ---------------------

def two_to_the_x_pwlin_vectorized_lut(x, inference_obj):
    """
    Approximates f(x) = 2^x
    - Input is 0 to -1 which is (16, 15)
    - Output is 1 to 1/2 which is (16, 15)
    """
    quant_in = 15
    quant_out = 15
    lut = [(-1, 0.5, 3.669850824967786),
           (-0.72750936, 0.6039456541218164, 3.901535238510212),
           (-0.47119999999999995, 0.7213643341679835, 4.132089343044075),
           (-0.22919168, 0.8531127439313528, 4.363160128674828),
           (0.00000000001, 1, 0)]
    lut_out = apply_pwlin(lut, x, inference_obj, quant_in, quant_out)
    return lut_out


def inverse_pwlin_vectorized_lut(x, inference_obj):
    """
    Approximates f(x) = 1/x
    - Input is 0.5 to 1 which is (16, 15)
    - Output is 1 to 2 which is (16, 14)
    """
    quant_in = 15
    quant_out = 14
    lut = [(0.5, 2.0, 10.570458022174277),
           (0.5946032800000001, 1.6817936154001705, 8.889148056938906),
           (0.7071000000000001, 1.414227124876255, 7.474024774299401),
           (0.84089672, 1.1892066840265472, 6.285225546575786),
           (1, 1.0, 0)]
    lut_out = apply_pwlin(lut, x, inference_obj, quant_in, quant_out)
    return lut_out


def gelu_pwlin_vectorized_lut(x, inference_obj):
    """
    Approximates f(x) = GELU(x)
    - Input is -3 to 3 which is (16, 8)
    - Output is -3 to 3 which is (16, 8)

    Note: this approximation is [-3, 3] which requires normalization beforehand
    """
    quant_in = 10
    quant_out = 8
    lut = [(-3, -0.0036373920817729943, 0.44438519308536634),
           (-0.7496999999999998, -0.17003910685389734, 2.547978758520282),
           (-0.3572320499999999, -0.12877242470687775, 2.799300902592587),
           (0.0, 0.0, 2.799300902592586),
           (0.35723205, 0.22845962529312222, 2.5479787585202818),
           (0.7497, 0.5796608931461028, 0.44438519308536634),
           (3.00001, 2.996362607918227, 0)]
    lut_out = apply_pwlin(lut, x, inference_obj, quant_in, quant_out)
    return lut_out

import src.config

def gelu_basic_pwlin_vectorized_lut(x, inference_obj, in_b, out_b):
    """
    Approximates f(x) = GELU(x)
    - Input is -128 to 127 which is (16, 8)
    - Output is -128 to 127 which is (16, 8)
    """
    quant_in = src.config.quan_in # in_b # 8
    quant_out = src.config.quan_out# 8 # out_b # 8
    lut = [(-128, -0.0, 0.008042410929059305),
           (-3.6591762320000214, -0.00032293305386205497, 0.34469491052090934),
           (-0.7580600000000146, -0.17003402799621162, 1.3019816160195725),
           (0.009999999999990905, 0.005039893559850952, 1.3576958272576445),
           (0.7465419999999909, 0.5765088167303702, 0.34288204042730686),
           (3.6629968797999912, 3.6626790770380957, 0.00810786685829746),
           (127.001, 127.0, 0)]
    lut_out = apply_pwlin(lut, x, inference_obj, quant_in, quant_out)
    return lut_out


def sqrt_inverse_pwlin_vectorized_lut(x, inference_obj):
    """
    Approximates f(x) = 1/sqrt(x)
    - Input is 0.25 to 1 which is (16, 15)
    - Output is 1 to 2 which is (16, 14)
    """
    quant_in = 15
    quant_out = 14
    lut = [(0.25, 2.0, 1 / (0.3028021504 - 0.25)),
           (0.3028021504, 1.8172744575887596, 1 / (0.364588 - 0.3028021504)),
           (0.364588, 1.6561467431164119, 1 / (0.43655929720000003 - 0.364588)),
           (0.43655929720000003, 1.5134858995467784, 1 / (0.52 - 0.43655929720000003)),
           (0.52, 1.3867504905630728, 1 / (0.727552 - 0.52)),
           (0.727552, 1.1723788648189515, 1 / (1.0 - 0.727552)),
           (1.0, 1.0, 0)]
    lut_out = apply_pwlin(lut, x, inference_obj, quant_in, quant_out)
    return lut_out


def split_bits_vectorized(array, word_len, frac_len):
    """
    Splits bits into integer and fraction for e^x approximation
    """
    v = Fxp(array, True, word_len, frac_len).bin()

    s = [x[0] for x in v]
    i = [x[1:-frac_len] for x in v]
    f = [x[-frac_len:] for x in v]

    i_negated = ["1b" + "".join(["0" if y == "1" else "1" for y in x]) for x in i]
    full_fraction = ["1b" + s[i] + f[i] for i in range(len(s))]
    full_fraction_float = Fxp(full_fraction, True, 9, 8).astype(float)
    shift_amount = Fxp(i_negated, False, word_len - frac_len - 1, 0).astype(float)
    shift_amount = np.array([shift_amount[i] if s[i] == "1" else 0 for i in range(len(shift_amount))])

    int_bits_to_keep = 4  # number of integer bits to keep (everything beyond is zeroed out) - 3 and 4 all can work
    val_okay = (shift_amount <= 2 ** int_bits_to_keep - 1)
    return val_okay, full_fraction_float, shift_amount


# ---------------- Kernel Approximations ---------------------

def pwlin_gelu(array, inference_obj):
    """
    GELU kernel, complex (with normalization)
    - Input = (16, 8)
    - Output = (16, 8)
    """
    q = array.copy()
    q[array > 3] = 0
    q[array < -3] = 0
    pwlin_out = gelu_pwlin_vectorized_lut(q, inference_obj)
    pwlin_out[array > 3] = array[array > 3]
    pwlin_out[array < -3] = 0
    return pwlin_out


def pwlin_gelu_basic(array, inference_obj, in_b, out_b):
    """
    GELU kernel, simple (direct pwlin approximation)
    - Input = (16, 8)
    - Output = (16, 8)
    """
    pwlin_out = gelu_basic_pwlin_vectorized_lut(array, inference_obj, in_b, out_b)
    return pwlin_out


def pwlin_softmax(array, inference_obj):
    """
    Softmax kernel
    - Input = (16, 9)
    - Output = (16, 15)
    """
    array = array * math.log2(math.e)

    array = inference_obj.quantize(array, Fxp(None, True, 16, 9)) # 8

    array = array - np.max(array)

    array = inference_obj.quantize(array, Fxp(None, True, 17, 9))# 8

    val_okay, full_fraction_float, shift_amount = split_bits_vectorized(array, 17, 9)# 8
    output_arr = np.zeros(array.shape)
    output_arr[val_okay] = two_to_the_x_pwlin_vectorized_lut(full_fraction_float[val_okay], inference_obj) * np.power(2,
                                                                                                                      -
                                                                                                                      shift_amount[
                                                                                                                          val_okay])
    # output_arr = Fxp(output_arr, True, 16, 15).astype(float)
    output_arr = inference_obj.quantize(output_arr, Fxp(None, True, 16, 15))  # can do (5, 4)
    # Technically because of max 4-bit shift we have (20, 19) but we can cut off last 4 fractional bits!
    sum_val = np.sum(output_arr)
    sum_bit_growth = int(math.log2(len(array)))
    sum_val = inference_obj.quantize(sum_val, Fxp(None, True, 16 + sum_bit_growth, 15))  # Can do 5+sum_bit_growth, 5
    # we have sum of 128 or 512 elements (7 or 9 bit growth) so we can go from (23/25, 15) to (16, 15-log2(len)=8/6)
    sum_val_inverse, clz = pwlin_inverse(np.array([sum_val]), inference_obj, sum_bit_growth)  # 1/sum_val
    sum_val_inverse[0] = inference_obj.quantize(sum_val_inverse[0], Fxp(None, True, 16,
                                                                        14 + sum_bit_growth))  # can do 5+sum_bit_growth, sum_bit_growth
    output_arr = output_arr * sum_val_inverse[0]
    output_arr = output_arr * np.power(2, clz[0])
    output_arr = inference_obj.quantize(output_arr, Fxp(None, True, 32, 15 + sum_bit_growth))
    return output_arr


def pwlin_inverse(array, inference_obj, sum_bit_growth):
    """
    Inverse kernel
    - Input = (16+7/9 = 23/25, 15)
    - Output = (16, 21 or 23)
    """
    sign = (array >= 0)  # save sign for later
    positive_array = np.abs(array)  # get positive value for clz
    bin_array = Fxp(positive_array, True, 16 + sum_bit_growth, 15).bin()
    clz = np.array([(x[1:-1] + "1").index("1") for x in
                    bin_array])  # count leading zeros excluding sign (assume last bit is 1, since all zeros is div by zero)
    array_shifted = positive_array * np.power(2, clz)

    array_shifted = array_shifted * (2 ** -sum_bit_growth)
    array_shifted = inference_obj.quantize(array_shifted, Fxp(None, True, 16, 15))  # problem! Decreases accuracy
    pwlin_out = inverse_pwlin_vectorized_lut(array_shifted, inference_obj)
    pwlin_out = inference_obj.quantize(pwlin_out, Fxp(None, True, 16, 14))  # since output is (1 to 2)
    pwlin_out = pwlin_out * (2 ** -sum_bit_growth)

    pwlin_out_shifted = pwlin_out  # * np.power(2, clz)
    return pwlin_out_shifted * (sign * 2 - 1), clz


def pwlin_sqrt_inverse(array, inference_obj):
    """
    Square root inverse kernel
    - Input = (35, 26)
    - Output = (16, 18) and clz = 0 to 17
    """
    sign = (array >= 0)  # save sign for later
    positive_array = np.abs(array)  # get positive value for clz
    bin_array = Fxp(positive_array, True, 35, 26).bin()
    clz = np.array([(x[1:-1] + "1").index("1") // 2 for x in
                    bin_array])  # count leading zeros excluding sign (assume last bit is 1, since all zeros is div by zero)
    array_shifted = positive_array * np.power(2, 2 * clz)  # shift to have all leading 0's

    array_shifted = array_shifted * (2 ** -8)  # interpret it as fully fractional signed 16-bit
    array_shifted = inference_obj.quantize(array_shifted, Fxp(None, True, 16, 15))
    pwlin_out = sqrt_inverse_pwlin_vectorized_lut(array_shifted, inference_obj)
    pwlin_out = inference_obj.quantize(pwlin_out, Fxp(None, True, 16, 14))  # since output is (1 to 2)
    pwlin_out = pwlin_out * (2 ** -4)  # undo fully fractional interpretation so it's (16, 21)

    pwlin_out_shifted = pwlin_out  # * np.power(2, clz)
    return pwlin_out_shifted * (sign * 2 - 1), clz


def pwlin_layer_norm(X, weights, inference_obj):
    """
    Layer normalization kernel
    - Input = (16, 6)
    - Output = (16, 8)
    """
    sum_val = np.sum(X, axis=1)
    sum_bit_growth = int(math.ceil(math.log2(X.shape[1])))  # 10 bits for sum
    sum_val = inference_obj.quantize(sum_val, Fxp(None, True, 16 + sum_bit_growth, 6))
    # Output is now (26, 6) because of 10 bits growth
    mean = sum_val / X.shape[1]  # this is actually 512+256 = 1100000000 so it's a shift-add
    # Output is now at (26, 16) = original with 10 extra fractional bits
    f1 = (X.T - mean).T
    # Difference is (27, 16) but we bring back to (26, 16) by chopping off edge case
    var = np.sum(f1 * f1, axis=1) / X.shape[1]
    # Products becomes (52, 32) shifted. Because we are multiplying by 1/sqrt(var) we can saturate some int bits safely?
    var = inference_obj.quantize(var, Fxp(None, True, 35, 26))  # 34, 26 = 7 int and 26 frac = edge case (used 35, 26)

    # Do square root inverse
    f2_inv, clz = pwlin_sqrt_inverse(var, inference_obj)  # 1/np.sqrt(var)

    normalized_output = (f1.T * f2_inv).T  # (26, 16) * (16, 21) = 42, 37
    normalized_output = inference_obj.quantize(normalized_output, Fxp(None, True, 42, 34))
    normalized_output = (normalized_output.T * np.power(2, clz)).T
    normalized_output = inference_obj.quantize(normalized_output, Fxp(None, True, 16,
                                                                      10))  # used to be 34, 27 but removed a bit then cut way down
    scaled_output = normalized_output * weights[0] + weights[1]
    return scaled_output
