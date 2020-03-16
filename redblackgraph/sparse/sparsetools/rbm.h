#ifndef __RBM_H__
#define __RBM_H__

inline template <class T, class U>
const T& avos_sum(const T& a, const T& b)
{
    if (a == 0 || (U)~b == 0) return b;
    if (b == 0 || (U)~a == 0) return a;
    if ((U)a < (U)b) return a;
    return b;
}

template <class T, class U> const short MSB(T x);
template <class T, class U> const U avos_product(const T& lhs, const T& rhs);
template <class I> void rbm_matmat_pass1(const I n_row, const I n_col, const I Ap[], const I Aj[], const I Bp[], const I Bj[], I Cp[]);
template <class I, class T, class U> void rbm_matmat_pass2(const I n_row, const I n_col, const I Ap[], const I Aj[], const T Ax[], const I Bp[], const I Bj[], const T Bx[], I Cp[], I Cj[], T Cx[]);

#endif
