#ifndef __RBM_H__
#define __RBM_H__

template <class T, class U> const short MSB(T x);
template <class T, class U> const U avos_product(const T& lhs, const T& rhs);
template <class T, class U> const T& avos_sum(const T& a, const T& b);
template <class I> void rbm_matmat_pass1(const I n_row, const I n_col, const I Ap[], const I Aj[], const I Bp[], const I Bj[], I Cp[]);
template <class I, class T, class U> void rbm_matmat_pass2(const I n_row, const I n_col, const I Ap[], const I Aj[], const T Ax[], const I Bp[], const I Bj[], const T Bx[], I Cp[], I Cj[], T Cx[]);

#endif
