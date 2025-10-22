#ifndef __RBM_MATH_H__
#define __RBM_MATH_H__

#include <set>
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>
#include <iostream>

// see: https://github.com/klmr/named-operator
// for how to define named operator in C++
template <class T, class U>
const short MSB(T x)
{
    if ((U)(~x) == 0) return 0;
    short targetlevel = 0;
    if ((U)~x == 0) {
        return targetlevel;
    }
    while (x >>= 1) {
        targetlevel += 1;
    }
    return targetlevel;
}

template <class T, class U>
const U avos_product(const T& lhs, const T& rhs)
{
    U x = (U)lhs;
    U y = (U)rhs;

    // zero property
    if (x == 0 || y == 0) {
        return 0;
    }
    // Special case -1 * 1 or -1 * -1
    // There is an oddity in bitwise NOT of an unsigned char, ~<unsigned char> temporary is an <int> rather
    // than an <unsigned char>. Because of this, cast ~x and ~y to the appropriate type
    if (lhs == (T)-1) {
        if (y == 1) {
            return x;
        }
        x = 1;
    }
    if (rhs == (T)-1) {
        if (x == 1) {
            return y;
        }
        y = 1;
    }

    short bit_position = MSB<T,U>(y);
    short result_size = MSB<T,U>(x) + bit_position;
    if (result_size >= (short)(sizeof(x) * 8)) {
        // Overflow Error
        PyErr_Format(PyExc_OverflowError,
                         "Avos product of %lu and %lu, results in an overflow. (Result size would require %u bits; Type provides %u bits)", \
                         lhs, rhs, result_size + 1, (short)(sizeof(x) * 8)
                         );
    }
    U result = ((y & (((U)1 << bit_position) - 1)) | (x << bit_position));
    if (result == (U)-1) {
        // Overflow Error
        PyErr_Format(PyExc_OverflowError,
        "Avos product of %lu and %lu, results in an overflow. Result of avos product collides with 'red' 1 (-1).", \
                                 lhs, rhs
        );
    }
    return result;
}

template <class T, class U>
const T& avos_sum(const T& a, const T& b)
{
    if (a == 0 || (U)(~b) == 0) return b;
    if (b == 0 || (U)(~a) == 0) return a;
    if ((U)a < (U)b) return a;
    return b;
}

/*
 * Pass 1 computes RBM row pointer for the matrix product C = A * B
 *
 */
template <class I>
void rbm_matmat_pass1(const I n_row,
                      const I n_col,
                      const I Ap[],
                      const I Aj[],
                      const I Bp[],
                      const I Bj[],
                            I Cp[])
{
    // method that uses O(n) temp storage
    std::vector<I> mask(n_col, -1);
    Cp[0] = 0;

    I nnz = 0;
    for(I i = 0; i < n_row; i++){
        npy_intp row_nnz = 0;

        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            I j = Aj[jj];
            for(I kk = Bp[j]; kk < Bp[j+1]; kk++){
                I k = Bj[kk];
                if(mask[k] != i){
                    mask[k] = i;
                    row_nnz++;
                }
            }
        }

        npy_intp next_nnz = nnz + row_nnz;

        if (row_nnz > NPY_MAX_INTP - nnz || next_nnz != (I)next_nnz) {
            /*
             * Index overflowed. Note that row_nnz <= n_col and cannot overflow
             */
            throw std::overflow_error("nnz of the result is too large");
        }

        nnz = next_nnz;
        Cp[i+1] = nnz;
    }
}


/*
 * Pass 2 computes RBM entries for matrix C = A*B using the
 * row pointer Cp[] computed in Pass 1.
 *
 */
template <class I, class T, class U>
void rbm_matmat_pass2(const I n_row,
      	              const I n_col,
      	              const I Ap[],
      	              const I Aj[],
      	              const T Ax[],
      	              const I Bp[],
      	              const I Bj[],
      	              const T Bx[],
      	                    I Cp[],
      	                    I Cj[],
      	                    T Cx[])
{
    // method that uses O(n) temp storage
    std::vector<I> next(n_col,-1);
    std::vector<T> sums(n_col, 0);
    Cp[0] = 0;

    I nnz = 0;
    for(I i = 0; i < n_row; i++){
        I head   = -2;
        I length =  0;

        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            I j = Aj[jj];
            T v = Ax[jj];

            for(I kk = Bp[j]; kk < Bp[j+1]; kk++){
                I k = Bj[kk];

                // change 1: redefinition of matrix multiplication, change + to <avos_sum> and * to <avos_product>
                sums[k] = avos_sum<T, U>(sums[k], avos_product<T, U>(v, Bx[kk]));

                if(next[k] == -1){
                    next[k] = head;
                    head  = k;
                    length++;
                }
            }
        }

        for(I jj = 0; jj < length; jj++){

            if(sums[head] != 0){
                Cj[nnz] = head;
                Cx[nnz] = sums[head];
                nnz++;
            }

            I temp = head;
            head = next[head];

            next[temp] = -1; //clear arrays
            sums[temp] =  0;
        }

        Cp[i+1] = nnz;
    }
}

#endif
