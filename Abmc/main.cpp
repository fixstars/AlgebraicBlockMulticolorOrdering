#include "Abmc.hpp"
#include "SymGS.hpp"

int main()
{
	Matrix A(N, N);

	for(auto i = decltype(n)(0); i < n; i++)
	{
		for(auto j = decltype(n)(0); j < n; j++)
		{
			const auto p = i*n + j;

			for(auto ii = ((i == 0) ? 0 : (i - 1)); ii <= ((i == n-1) ? (n - 1) : (i + 1)); ii++)
			{
				for(auto jj = ((j == 0) ? 0 : (j - 1)); jj <= ((j == n - 1) ? (n - 1) : (j + 1)); jj++)
				{
					const auto q = ii*n + jj;

					const auto value = (p == q) ? 8 : -1;
					A(p, q) = value;
				}
			}
		}
	}

	// std::cout << "元行列" << std::endl;
	// for(auto i = decltype(N)(0); i < N; i++)
	// {
	// 	for(auto j = decltype(N)(0); j < N; j++)
	// 	{
	// 		std::cout << A(i, j) << ", ";
	// 	}
	// 	std::cout << std::endl;
	// }
	std::cout << "####################################" << std::endl;

	Vector b(N);
	Vector expect(N);
	std::fill_n(expect.begin(), N, 100.1);
	b = boost::numeric::ublas::prod(A, expect);

	// 逐次
	Jacobi(A, b, expect); std::cout << "####################################" << std::endl;
	GaussSeidel(A, b, expect); std::cout << "####################################" << std::endl;
	GaussSeidel2(A, b, expect); std::cout << "####################################" << std::endl;
	SymmetryGaussSeidel(A, b, expect); std::cout << "####################################" << std::endl;

	// // 多色順序順序付け
	// GeometicMultiColoring(A, b, expect); std::cout << "####################################" << std::endl;
	// AlgebraicMultiColoring(A, b, expect); std::cout << "####################################" << std::endl;

	// // 2x2にブロック化
	// GeometicBlockMultiColoring<2>(A, b, expect); std::cout << "####################################" << std::endl;
	// AlgebraicBlockMultiColoring<2>(A, b, expect); std::cout << "####################################" << std::endl;

	// // 4x4にブロック化
	// GeometicBlockMultiColoring<4>(A, b, expect); std::cout << "####################################" << std::endl;
	// AlgebraicBlockMultiColoring<4>(A, b, expect); std::cout << "####################################" << std::endl;

	// // Cuthill-Mckee
	// CuthillMckee(A, b, expect); std::cout << "####################################" << std::endl;

	return 0;
}
