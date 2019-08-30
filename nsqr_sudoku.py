import numpy as np
import copy
from functools import partial
from itertools import chain

############
# Given
############

def _split2D(M, bl):
    for Mi in np.split(M, bl, axis=0):
        for Mj in np.split(Mi, bl, axis=1):
            yield Mj
def _count(A):
    return {k:v for k, v in zip(*np.unique(A, return_counts=True)) if k>0}
            
def _isrepeated(A):
    counts = _count(A)
    if not counts or max(counts.values()) <= 1:
        return False
    return True

def _shape_info(M):
    shape = M.shape
    length = np.mean(M.shape)
    block_length = np.sqrt(length)      
    assert len(shape) == 2
    assert not np.std(shape)
    assert not block_length % 1
    
    return shape, int(length), int(block_length)

def _get_block(M, i, j):
    shape, length, block_length = _shape_info(M)
    
    i0 = (i // block_length) * block_length
    i1 = i0 + block_length
    j0 = (j // block_length) * block_length
    j1 = j0 + block_length
    i0, i1, j0, j1 = map(int, [i0, i1, j0, j1])
    return M[i0:i1, j0:j1]
        
def isvalid(M):
    try:
        shape, length, block_length = _shape_info(M)
        
        assert np.max(M) >= 0
        assert np.max(M) <= length
        assert not any(np.apply_along_axis(_isrepeated, 0, M))
        assert not any(np.apply_along_axis(_isrepeated, 1, M))
        assert not any(list(map(_isrepeated, _split2D(M, block_length))))

    except AssertionError:
        return False
    return True

def find_candidate(M, i, j):    
    shape, length, block_length = _shape_info(M)
    candidates = set(range(1, length+1))
    
    v_col = M[i, :]
    v_col[j] = 0
    candidates -= set(v_col)
    
    v_row = M[:, j]
    v_row[i] = 0
    candidates -= set(v_row)

    v_block = _get_block(M, i, j)
    v_block[i%block_length, j%block_length] = 0
    candidates -= set(v_block.reshape(-1))

    return candidates

############
# Solution
############

def _fill(v, M, i, j):
    ret = copy.deepcopy(M)
    ret[i, j] = v
    return ret

def _find_target(M):
    for i, j in zip(*np.where(M == 0)):
        return i, j

def _step(M):
    if not 0 in M:
        return []
    i, j = _find_target(M)
    candidates = find_candidate(M, i, j)
    if not candidates:
        return []
    _p_fill = partial(_fill, M=M, i=i, j=j)
    Ms = list(map(_p_fill, candidates))
    return Ms


def _process(Ms):

    for M in Ms:
        if not isvalid(M):
            continue
        if not 0 in M:
            yield M
        Ms = _step(M)
        yield from _process(Ms)

def main(M):
    yield from _process([M])
        
if __name__ == '__main__':

    M = np.array([[5,3,0,0,7,0,0,0,0],
                  [6,0,0,1,9,5,0,0,0],
                  [0,9,8,0,0,0,0,6,0],
                  [8,0,0,0,6,0,0,0,3],
                  [4,0,0,8,0,3,0,0,1],
                  [7,0,0,0,2,0,0,0,6],
                  [0,6,0,0,0,0,2,8,0],
                  [0,0,0,4,1,9,0,0,5],
                  [0,0,0,0,8,0,0,7,9]])

    if not isvalid(M):
        print('illigle input')
    else:
        for i in main(M):
            print(i)
            break # commit this line for get all soltuions
        else:
            print('unsolvable!')
