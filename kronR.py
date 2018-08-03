"""Functions to construct sparse matrices
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import bsr_matrix
import sys
import psutil
from globalpara import *
import time

def kronR(A, B, format=None):
    """kronecker product of sparse matrices A and B
    Parameters
    ----------
    A : sparse or dense matrix
        first matrix of the product
    B : sparse or dense matrix
        second matrix of the product
    format : str, optional
        format of the result (e.g. "csr")
    Returns
    -------
    kronecker product in a sparse matrix format
    Examples
    --------
    >>> from scipy import sparse
    >>> A = sparse.csr_matrix(np.array([[0, 2], [5, 0]]))
    >>> B = sparse.csr_matrix(np.array([[1, 2], [3, 4]]))
    >>> sparse.kron(A, B).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])
    >>> sparse.kron(A, [[1, 2], [3, 4]]).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])
    """
    B = coo_matrix(B)
    mry0 = psutil.virtual_memory().used/1024.0**3
    ts = time.time()
    if (format is None or format == "bsr") and 2*B.nnz >= B.shape[0] * B.shape[1]:
        # B is fairly dense, use BSR
        A = csr_matrix(A,copy=True)

        output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

        if A.nnz == 0 or B.nnz == 0:
            # kronecker product is the zero matrix
            return coo_matrix(output_shape)

        B = B.toarray()
        data = A.data.repeat(B.size).reshape(-1,B.shape[0],B.shape[1])
        data = data * B

        return bsr_matrix((data,A.indices,A.indptr), shape=output_shape)
    else:
        # use COO
        print 'Size of AB =', (sys.getsizeof(A)+sys.getsizeof(B))/1024.0/1024.0/1024.0, 'G'
        A = coo_matrix(A)
        output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

        if A.nnz == 0 or B.nnz == 0:
            # kronecker product is the zero matrix
            return coo_matrix(output_shape)

        # expand entries of a into blocks
        print 'start expand'
        row = A.row.repeat(B.nnz)
        col = A.col.repeat(B.nnz)
        data = A.data.repeat(B.nnz)
        print 'size of rc =', \
              (sys.getsizeof(row)+sys.getsizeof(col))/1024.0/1024.0/1024.0, 'G'
        print '  size of d =', \
              (sys.getsizeof(data))/1024.0/1024.0/1024.0, 'G'
        tnote(ts, mry0, 'expand = ')
        print ''

        print 'start multiply'
        row *= B.shape[0]
        col *= B.shape[1]
        print 'size of rc =', \
              (sys.getsizeof(row)+sys.getsizeof(col))/1024.0/1024.0/1024.0, 'G'
        tnote(ts, mry0, 'multiply= ')
        print ''

        # increment block indices
        print 'start increment'
        row,col = row.reshape(-1,B.nnz),col.reshape(-1,B.nnz)
        print ''
        print '1 size of rc =', \
              (sys.getsizeof(row)+sys.getsizeof(col))/1024.0/1024.0/1024.0, 'G'
        row += B.row
        col += B.col
        print ''
        print '2 size of rc =', \
              (sys.getsizeof(row)+sys.getsizeof(col))/1024.0/1024.0/1024.0, 'G'
        row,col = row.reshape(-1),col.reshape(-1)
        print ''
        print '3 size of rc =', \
              (sys.getsizeof(row)+sys.getsizeof(col))/1024.0/1024.0/1024.0, 'G'
        tnote(ts, mry0, 'increment= ')
        print ''

        # compute block entries
        print 'start block entries'
        data = data.reshape(-1,B.nnz) * B.data
        data = data.reshape(-1)
        print '  size of d =', \
              (sys.getsizeof(data))/1024.0/1024.0/1024.0, 'G'
        tnote(ts, mry0, 'block entry = ')
        print ''

    return coo_matrix((data,(row,col)), shape=output_shape).asformat(format)
