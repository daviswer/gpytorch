from .. import libfft

def fft1(input):
    # [..., d]
    orig_size = input.size()
    orig_type = type(input)

    input = input.view(-1, input.size(-1))
    n, d = input.size()

    output = input.new().resize_(n, (d // 2) + 1, 2)
    if input.is_cuda:
        libfft.fft1_r2c_cuda(input, output)
    else:
        assert False

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
    
    output = input.new().resize_(nPlanes, n, (d // 2) + 1, 2)
    if input.is_cuda:
        libfft.fft2_r2c_cuda(input, output)
    else:
        assert False
    
    if len(orig_size) > 2:
        output_size = list(orig_size[:-2]) + [n, (d // 2) + 1, 2]
    else:
        output_size = [n, (d // 2) + 1, 2]
    return output.view(*output_size).type(orig_type)

def fft3(input):
    # [..., m n d]
    orig_size = input.size()
    orig_type = type(input)
    
    input = input.view(-1, input.size(-3), input.size(-2), input.size(-1))
    nPlanes, m, n, d = input.size()
    
    output = input.new().resize_(nPlanes, m, n, (d // 2) + 1, 2)
    if input.is_cuda:
        libfft.fft3_r2c_cuda(input, output)
    else:
        assert False
    
    if len(orig_size) > 3:
        output_size = list(orig_size[:-3]) + [m, n, (d // 2) + 1, 2]
    else:
        output_size = [m, n, (d // 2) + 1, 2]
    return output.view(*output_size).type(orig_type)

def fftc(input):
    # [..., m n d]
    orig_size = input.size()
    orig_type = type(input)
    
    input = input.view(-1, input.size(-3), input.size(-2), input.size(-1))
    nPlanes, m, n, d = input.size()
    
    output = input.new().resize_(nPlanes, (m // 2) + 1, n, d, 2)
    if input.is_cuda:
        libfft.fftc_r2c_cuda(input, output)
    else:
        assert False
    
    if len(orig_size) > 3:
        output_size = list(orig_size[:-3]) + [(m // 2) + 1, n, d, 2]
    else:
        output_size = [(m // 2) + 1, n, d, 2]
    return output.view(*output_size).type(orig_type)

def fft2_c(input):
    # [..., n d 2]
    orig_size = input.size()
    orig_type = type(input)
    
    input = input.view(-1, input.size(-3), input.size(-2), 2)
    nPlanes, n, d, _ = input.size()
    
    output = input.new().resize_(nPlanes,n,d,2)
    if input.is_cuda:
        libfft.fft2_c2c_cuda(input, output)
    else:
        assert False
    
    if len(orig_size) > 2:
        output_size = list(orig_size[:-3]) + [n, d, 2]
    else:
        output_size = [n, d, 2]
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
        assert False
#     output.div_(d)
    return output.view(size).type(orig_type)

def ifft2(input, size=None):
    # [..., n, d, 2]
    orig_type = type(input)

    if not size:
        size = list(input.size())[:-1]
        d = (size[-1] - 1) * 2
        size[-1] = d
        n = size[-2]
    else:
        d = size[-1]
        n = size[-2]
    input = input.view(-1, *input.size()[-3:])

    output = input.new().resize_(input.size(0), n, d)
    if input.is_cuda:
        libfft.fft2_c2r_cuda(input, output)
    else:
        assert False
    output.div_(n*d)
    return output.view(size).type(orig_type)

def ifft3(input, size=None):
    # [..., m, n, d, 2]
    orig_type = type(input)

    if not size:
        size = list(input.size())[:-1]
        d = (size[-1] - 1) * 2
        size[-1] = d
        n = size[-2]
        m = size[-3]
    else:
        d = size[-1]
        n = size[-2]
        m = size[-3]
    input = input.view(-1, *input.size()[-4:])

    output = input.new().resize_(input.size(0), m, n, d)
    if input.is_cuda:
        libfft.fft3_c2r_cuda(input, output)
    else:
        assert False
    output.div_(m*n*d)
    return output.view(size).type(orig_type)

def ifftc(input, size=None):
    # [..., m, n, d, 2]
    orig_type = type(input)

    if not size:
        size = list(input.size())[:-1]
        d = size[-1]
        n = size[-2]
        m = (size[-3] - 1) * 2
        size[-3] = m
    else:
        d = size[-1]
        n = size[-2]
        m = size[-3]
    input = input.view(-1, *input.size()[-4:])

    output = input.new().resize_(input.size(0), m, n, d)
    if input.is_cuda:
        libfft.fftc_c2r_cuda(input, output)
    else:
        assert False
    output.div_(m)
    return output.view(size).type(orig_type)

def ifft2_c(input):
    # [..., n d 2]
    orig_size = input.size()
    orig_type = type(input)
    
    input = input.view(-1, input.size(-3), input.size(-2), 2)
    nPlanes, n, d, _ = input.size()
    
    output = input.new().resize_(nPlanes,n,d,2)
    if input.is_cuda:
        libfft.ifft2_c2c_cuda(input, output)
    else:
        assert False
    
    if len(orig_size) > 2:
        output_size = list(orig_size[:-3]) + [n, d, 2]
    else:
        output_size = [n, d, 2]
    output.div_(n*d)
    return output.view(*output_size).type(orig_type)

def cmul(input,output):
    assert type(input)==type(output)
    orig_size = input.size()
    orig_type = type(input)
    
    input = input.view(-1,2)
    output = output.view(-1,2)
    
    assert input.size(0)==output.size(0)
    
    if input.is_cuda:
        libfft.cmul_cuda(input, output)
    else:
        assert False
        
    return output.view(*orig_size).type(orig_type)
