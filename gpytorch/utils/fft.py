from .. import libfft
from math import sqrt

def fft1(input):
    # [..., d]
    orig_size = input.size()
    orig_type = type(input)

    input = input.view(-1, input.size(-1))
    input.div_(sqrt(d))
    n, d = input.size()

    output = input.new().resize_(n, (d // 2) + 1, 2)
    if input.is_cuda:
        libfft.fft1_r2c_cuda(input, output)
    else:
        output = output.float()
        libfft.fft1_r2c(input.float(), output)

    if len(orig_size) > 1:
        output_size = list(orig_size[:-1]) + [(d // 2) + 1, 2]
    else:
        output_size = [(d // 2) + 1, 2]
    return output.view(*output_size).type(orig_type)

def fft2(input):
    # [..., n d]
    orig_size = input.size()
    orig_type = type(input)
    
    input = input.view(-1, input.size(-2), input.size(-1))
    nPlanes, n, d = input.size()
    
    output = input.new().resize_(nPlanes, (n // 2) + 1, (d // 2) + 1, 2)
    if input.is_cuda:
        libfft.fft2_r2c_cuda(input, output)
    else:
        assert False
    
    if len(orig_size) > 2:
        output_size = list(orig_size[:-2]) + [(n // 2) + 1, (d // 2) + 1, 2]
    else:
        output_size = [(n // 2) + 1, (d // 2) + 1, 2]
    output.div_(sqrt(n*d))
    return output.view(*output_size).type(orig_type)

def ifft1(input, size=None):
    # [..., d, 2]
    orig_type = type(input)

    if not size:
        size = list(input.size())[:-1]
        d = (size[-1] - 1) * 2
        size[-1] = d
    else:
        d = size[-1]
    input = input.view(-1, *input.size()[-2:])

    output = input.new().resize_(input.size(0), d)
    if input.is_cuda:
        libfft.fft1_c2r_cuda(input, output)
    else:
        output = output.float()
        libfft.fft1_c2r(input.float(), output)
    output.div_(sqrt(d))
    return output.view(size).type(orig_type)

def ifft2(input, size=None):
    # [..., n, d, 2]
    orig_type = type(input)

    if not size:
        size = list(input.size())[:-1]
        d = (size[-1] - 1) * 2
        size[-1] = d
        n = (size[-2] -1) * 2
        size[-2] = n
    else:
        d = size[-1]
        n = size[-2]
    input = input.view(-1, *input.size()[-3:])

    output = input.new().resize_(input.size(0), n, d)
    if input.is_cuda:
        libfft.fft2_c2r_cuda(input, output)
    else:
        assert False
    output.div_(sqrt(n*d))
    return output.view(size).type(orig_type)
