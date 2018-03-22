#ifndef COMMON_HPP_INCLUDED
#define COMMON_HPP_INCLUDED

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>

static constexpr std::size_t n = 8;
static constexpr std::size_t N = n*n;

using Matrix = boost::numeric::ublas::compressed_matrix<double>;
using Vector = boost::numeric::ublas::vector<double>;
using Index = std::remove_const_t<decltype(N)>;
using Color = Index;
using Block = Index;
using Level = Index;

#endif
