#ifndef SYMGS_HPP_INCLUDED
#define SYMGS_HPP_INCLUDED
#include <iostream>

#include "common.hpp"

// 普通のガウスザイデル法
static void GaussSeidel(const Matrix& A, const Vector& b, const Vector& expect)
{
	Vector x(N);
	std::fill_n(x.begin(), N, 0);
	// 初期残差
	{
		const auto r = expect - x;
		const auto rr = boost::numeric::ublas::inner_prod(r, r);
		std::cout << 0 << ", " << rr << std::endl;
	}
	for(auto iteration = decltype(N)(0); iteration < N; iteration++)
	{
		for(auto i = decltype(N)(0); i < N; i++)
		{
			auto x_i = b(i);

			const auto offset = A.index1_data()[i];
			const auto count = A.index1_data()[i + 1] - offset;
			double a_ii;
			for(auto idx = decltype(count)(0); idx < count; idx++)
			{
				const auto j = A.index2_data()[offset + idx];
				const double a_ij = A.value_data()[offset + idx];

				if(j == i)
				{
					a_ii = a_ij;
				}
				else
				{
					const double x_j = x[j];
					x_i -= a_ij * x_j;
				}
			}
			x(i) = x_i / a_ii;
		}

		const auto r = expect - x;
		const auto rr = boost::numeric::ublas::inner_prod(r, r);
		std::cout << iteration+1 << ", " << rr << std::endl;
	}
}

#endif
