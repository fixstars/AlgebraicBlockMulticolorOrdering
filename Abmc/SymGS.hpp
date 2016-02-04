#ifndef SYMGS_HPP_INCLUDED
#define SYMGS_HPP_INCLUDED
#include <iostream>

#include "common.hpp"

static void GaussSeidelMain(Vector& x, const Matrix& A, const Vector& b, const std::size_t i)
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

template<typename SolveFunction>
static void Solve(const std::string name, const Vector& expect, SolveFunction solve)
{
	std::cout << name << std::endl;
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
		solve(x);

		const auto r = expect - x;
		const auto rr = boost::numeric::ublas::inner_prod(r, r);
		std::cout << iteration + 1 << ", " << rr << std::endl;
	}
}

// 普通のガウスザイデル法
static void GaussSeidel(const Matrix& A, const Vector& b, const Vector& expect)
{
	Solve("ガウスザイデル法", expect, [&A, &b](Vector& x)
	{
		for(auto i = decltype(N)(0); i < N; i++)
		{
			GaussSeidelMain(x, A, b, i);
		}
	});
}

// 対称ガウスザイデル法
static void SymmetryGaussSeidel(const Matrix& A, const Vector& b, const Vector& expect)
{
	Solve("対称ガウスザイデル法", expect, [&A, &b](Vector& x)
	{
		// 順
		for(auto i = decltype(N)(0); i < N; i++)
		{
			GaussSeidelMain(x, A, b, i);
		}

		// 逆順
		for(auto i = std::make_signed_t<decltype(N)>(N - 1); i > 0; i--)
		{
			GaussSeidelMain(x, A, b, i);
		}
	});
}

#endif
