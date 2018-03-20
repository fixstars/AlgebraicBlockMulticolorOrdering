#ifndef SYMGS_HPP_INCLUDED
#define SYMGS_HPP_INCLUDED
#include <iostream>

#include "common.hpp"

static void JacobiMain(Vector& x, const Matrix& A, const Vector& b, const Vector& x_old, const std::size_t i)
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
			const double x_j = x_old[j];
			x_i -= a_ij * x_j;
		}
	}
	x(i) = x_i / a_ii;
}

static void GaussSeidelMain(Vector& x, const Matrix& A, const Vector& b,const std::size_t i)
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
	std::cout << "+++++++++++++++++++++" << std::endl;
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

// 普通のヤコビ法
static void Jacobi(const Matrix& A, const Vector& b, const Vector& expect)
{
	Vector x_old(b.size());
	Solve("ヤコビ法", expect, [&A, &b, &x_old](Vector& x)
	{
		std::copy(x.cbegin(), x.cend(), x_old.begin());
		for(auto i = decltype(N)(0); i < N; i++)
		{
			JacobiMain(x, A, b, x_old, i);
		}
	});
}

// ガウスザイデル法
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

// 普通のガウスザイデル法を2回
static void GaussSeidel2(const Matrix& A, const Vector& b, const Vector& expect)
{
	Solve("ガウスザイデル法x2", expect, [&A, &b](Vector& x)
	{
		for(auto i = decltype(N)(0); i < N; i++)
		{
			GaussSeidelMain(x, A, b, i);
		}
		for(auto i = decltype(N)(0); i < N; i++)
		{
			GaussSeidelMain(x, A, b, i);
		}
	});
}

// 多色順序付けのガウスザイデル法
static void GaussSeidel(const Matrix& A, const Vector& b, const Vector& expect,
	const Index row[],
	const Index offset[],
	const Color colorCount)
{
	Solve("多色順序付けガウスザイデル法", expect, [&A, &b, &row, &offset, colorCount](Vector& x)
	{
		for (auto color = decltype(colorCount)(0); color < colorCount; color++)
		{
			for (auto idx = offset[color]; idx < offset[color + 1]; idx++)
			{
				const auto i = row[idx];
				GaussSeidelMain(x, A, b, i);
			}
		}
	});
}

// ブロック化多色順序付けのガウスザイデル法
static void GaussSeidel(const Matrix& A, const Vector& b, const Vector& expect,
	const Index row[],
	const Block blockOffset[],
	const Index offset[],
	const Color colorCount)
{
	Solve("ブロック化多色順序付けガウスザイデル法", expect, [&A, &b, &row, &blockOffset, &offset, colorCount](Vector& x)
	{
		// 順
		for (auto color = decltype(colorCount)(0); color < colorCount; color++)
		{
			for (auto block = blockOffset[color]; block < blockOffset[color + 1]; block++)
			{
				for (auto idx = offset[block]; idx < offset[block + 1]; idx++)
				{
					const auto i = row[idx];
					GaussSeidelMain(x, A, b, i);
				}
			}
		}
	});
}

// CutHill-Mckeeのガウスザイデル法
static void GaussSeidelForCuthillMckee(const Matrix& A, const Vector& b, const Vector& expect,
	const Index row[],
	const Index offset[],
	const Color levelConut)
{
	Solve("CutHill-Mckeeガウスザイデル法", expect, [&A, &b, &row, &offset, levelConut](Vector& x)
	{
		// 順
		for (auto level = decltype(levelConut)(0); level < levelConut; level++)
		{
			for (auto idx = offset[level]; idx < offset[level + 1]; idx++)
			{
				const auto i = row[idx];
				GaussSeidelMain(x, A, b, i);
			}
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
		for(auto i = static_cast<std::make_signed_t<decltype(N)>>(N - 1); i >= 0; i--)
		{
			GaussSeidelMain(x, A, b, i);
		}
	});
}

// 多色順序付けの対称ガウスザイデル法
static void SymmetryGaussSeidel(const Matrix& A, const Vector& b, const Vector& expect,
	const Index row[],
	const Index offset[],
	const Color colorCount)
{
	Solve("多色順序付け対称ガウスザイデル法", expect, [&A, &b, &row, &offset, colorCount](Vector& x)
	{
		// 順
		for(auto color = decltype(colorCount)(0); color < colorCount; color++)
		{
			for(auto idx = offset[color]; idx < offset[color+1]; idx++)
			{
				const auto i = row[idx];
				GaussSeidelMain(x, A, b, i);
			}
		}

		// 逆順
		for(auto color = static_cast<std::make_signed_t<decltype(colorCount)>>(colorCount - 1); color >= 0; color--)
		{
			for(auto idx = offset[color]; idx < offset[color + 1]; idx++) // 同じ色の中では点間の依存関係はないのでここは逆順にする必要がない
			{
				const auto i = row[idx];
				GaussSeidelMain(x, A, b, i);
			}
		}
	});
}

// ブロック化多色順序付けの対称ガウスザイデル法
static void SymmetryGaussSeidel(const Matrix& A, const Vector& b, const Vector& expect,
	const Index row[],
	const Block blockOffset[],
	const Index offset[],
	const Color colorCount)
{
	Solve("ブロック化多色順序付け対称ガウスザイデル法", expect, [&A, &b, &row, &blockOffset, &offset, colorCount](Vector& x)
	{
		// 順
		for(auto color = decltype(colorCount)(0); color < colorCount; color++)
		{
			for(auto block = blockOffset[color]; block < blockOffset[color + 1]; block++)
			{
				for(auto idx = offset[block]; idx < offset[block + 1]; idx++)
				{
					const auto i = row[idx];
					GaussSeidelMain(x, A, b, i);
				}
			}
		}

		// 逆順
		for(auto color = static_cast<std::make_signed_t<decltype(colorCount)>>(colorCount - 1); color >= 0; color--)
		{
			for(auto block = blockOffset[color]; block < blockOffset[color + 1]; block++) // 同じ色の中ではブロック間に依存関係はないのでここは逆順にする必要がない
			{
				using SignedIndex = std::make_signed_t<decltype(offset[0])>;
				const auto first = static_cast<SignedIndex>(offset[block + 1]) - 1;
				const auto last = static_cast<SignedIndex>(offset[block]);
				for(auto idx = first; idx >= last; idx--)
				{
					const auto i = row[idx];
					GaussSeidelMain(x, A, b, i);
				}
			}
		}
	});
}

// CutHill-Mckeeのガウスザイデル法
static void SymmetryGaussSeidelForCutHillMckee(const Matrix& A, const Vector& b, const Vector& expect,
	const Index row[],
	const Index offset[],
	const Level levelCount)
{
	Solve("CutHill-Mckee対称ガウスザイデル法", expect, [&A, &b, &row, &offset, levelCount](Vector& x)
	{
		// 順
		for(auto level = decltype(levelCount)(0); level < levelCount; level++)
		{
			for(auto idx = offset[level]; idx < offset[level + 1]; idx++)
			{
				const auto i = row[idx];
				GaussSeidelMain(x, A, b, i);
			}
		}

		// 逆順
		for(auto level = static_cast<std::make_signed_t<decltype(levelCount)>>(levelCount - 1); level > 0; level--)
		{
			for(auto idx = offset[level+1] - 1; idx >= offset[level]; idx--) // 隣接ノードの排除をしていないので、同じLevel内でも逆順にする必要がある。
			{
				const auto i = row[idx];
				GaussSeidelMain(x, A, b, i);
			}
		}
	});
}

#endif
