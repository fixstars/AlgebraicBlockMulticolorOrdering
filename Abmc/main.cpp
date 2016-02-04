#include <iostream>
#include <algorithm>
#include <queue>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>

static constexpr std::size_t n = 8;
static constexpr std::size_t N = n*n;

using Matrix = boost::numeric::ublas::compressed_matrix<double>;
using Index = std::remove_const_t<decltype(N)>;
using Color = Index;
using Block = Index;

static constexpr std::size_t MAX_COLOR_COUNT = 27;
static constexpr std::size_t BLOCK_SIZE = 2; // 2x2にブロック化

//#define ENABLE_ROW_DATA

static void OutputResult(const std::string name, const Matrix& A, const Index row[], const Color color[])
{
	// 元の行番号→並び替え後の行番号の変換表
	auto lut = std::make_unique<Index[]>(N);
	{
		// 並び替え後の行番号順に
		auto pair = std::make_unique<std::pair<Index, Index>[]>(N);
		for(auto i = decltype(N)(0); i < N; i++)
		{
			pair[i].first = row[i];
			pair[i].second = i;
		}
		std::sort(pair.get(), pair.get() + N, [](auto left, auto right)
		{
			return (left.first < right.first);
		});

		// 変換表
		std::transform(pair.get(), pair.get() + N, lut.get(), [](const auto p)
		{
			return p.second;
		});
	}

	Matrix B(N, N);
	for(auto i = decltype(N)(0); i < N; i++)
	{
		const auto r = row[i];

		const auto offset = A.index1_data()[r];
		const auto count = A.index1_data()[r + 1] - offset;

		// 接続している点に使われている色を洗い出す
		for(auto idx = decltype(count)(0); idx < count; idx++)
		{
			const auto j = A.index2_data()[offset + idx];
			const auto value = A.value_data()[offset + idx];

			const auto c = lut[j];
			B(i, c) = value;
		}
	}

	std::cout << name << std::endl;
	for(auto i = decltype(N)(0); i < N; i++)
	{
		for(auto j = decltype(N)(0); j < N; j++)
		{
			std::cout << B(i, j) << ", ";
		}
		std::cout << std::endl;
	}

#ifndef ENABLE_ROW_DATA
	for(auto i = decltype(n)(0); i < n; i++)
	{
		for(auto j = decltype(n)(0); j < n; j++)
		{
			std::cout << color[i*n + j] << ", ";
		}
		std::cout << std::endl;
	}
#endif
}

// （非ブロック化）多色順序付けの行番号配列を生成
static void CreateRow(Index row[], const Index color[])
{
	// 色番号の小さい順に並び替え（同じ色の場合は行番号の小さい順）
	auto pair = std::make_unique<std::pair<Color, Index>[]>(N);
	for(auto i = decltype(N)(0); i < N; i++)
	{
		pair[i].first = color[i];
		pair[i].second = i;
	}
	std::sort(pair.get(), pair.get() + N, [](auto left, auto right)
	{
		return (left.first != right.first) ? (left.first < right.first) : (left.second < right.second);
	});

	// 行番号だけの配列に変換
	std::transform(pair.get(), pair.get() + N, row, [](const auto p)
	{
		return p.second;
	});
}

// 幾何形状を用いた多色順序付け
static void GeometicMultiColoring(const Matrix& A)
{
	auto color = std::make_unique<Color[]>(N);

	for(auto i = decltype(n)(0); i < n; i++)
	{
		for(auto j = decltype(n)(0); j < n; j++)
		{
			const auto idx = i*n + j;

			const auto isEvenRow    = i % 2 == 0;
			const auto isEvenColumn = j % 2 == 0;

			color[idx] = isEvenRow ?
				(isEvenColumn ?
					Color(1) :  // 偶数行の偶数列
					Color(2)) : // 偶数行の奇数列
				(isEvenColumn ?
					Color(3) :  // 奇数業の偶数列
					Color(4));  // 奇数業の奇数列
		}
	}

	// 並び替え後の行番号→元の行番号の変換表
	auto row = std::make_unique<Index[]>(N);
	CreateRow(row.get(), color.get());

	OutputResult("幾何的多色順序付け", A, row.get(), color.get());
}

static decltype(auto) AlgebraicMultiColoringMain(const Matrix& A)
{
	const auto size = A.size1();

	auto inNeighbor = std::array<bool, MAX_COLOR_COUNT>();
	constexpr Color INVALID_COLOR = 0;
	auto color = std::vector<Color>(size, INVALID_COLOR);
	color[0] = Color(1); // 最初の点は色1

	for(auto i = decltype(size)(1); i < size; i++)
	{
		const auto offset = A.index1_data()[i];
		const auto count = A.index1_data()[i + 1] - offset;

		// 接続している点に使われている色を洗い出す
		std::fill_n(inNeighbor.begin(), MAX_COLOR_COUNT, false);
		for(auto idx = decltype(count)(0); idx < count; idx++)
		{
			const auto j = A.index2_data()[offset + idx];
			const auto color_j = color[j];
			if(color_j != INVALID_COLOR)
			{
				inNeighbor[color_j] = true;
			}
		}

		// 使われていない色のうち最小値を自分の色にする
		for(Color candiate = 1; candiate < MAX_COLOR_COUNT; candiate++)
		{
			if(!inNeighbor[candiate])
			{
				color[i] = candiate;
				break;
			}
		}
	}

	return color;
}

// 幾何形状を用いない多色順序付け
static void AlgebraicMultiColoring(const Matrix& A)
{
	const auto&& color = AlgebraicMultiColoringMain(A);

	// 並び替え後の行番号→元の行番号の変換表
	auto row = std::make_unique<Index[]>(N);
	CreateRow(row.get(), color.data());

	OutputResult("代数的多色順序付け", A, row.get(), color.data());
}

// （非ブロック化）多色順序付けの行番号配列を生成
static void CreateRow(Index row[], const Index color[], const Block block[])
{
	// 色番号の小さい順に並び替え
	// ・同じ色の場合はブロック番号の小さい順
	// ・同じブロックなら行番号の小さい順
	auto data = std::make_unique<std::tuple<Color, Block, Index>[]>(N);
	for(auto i = decltype(N)(0); i < N; i++)
	{
		data[i] = std::make_tuple(color[i], block[i], i);
	}
	std::sort(data.get(), data.get() + N, [](auto left, auto right)
	{
		{
			const auto leftColor = std::get<0>(left); const auto rightColor = std::get<0>(right);
			if(leftColor != rightColor)
			{
				return leftColor < rightColor;
			}
		}
		{
			const auto leftBlock = std::get<1>(left); const auto rightBlock = std::get<1>(right);
			if(leftBlock != rightBlock)
			{
				return leftBlock < rightBlock;
			}
		}
		{
			const auto leftIndex = std::get<2>(left); const auto rightIndex = std::get<2>(right);
			{
				return leftIndex < rightIndex;
			}
		}
	});

	// 行番号だけの配列に変換
	std::transform(data.get(), data.get() + N, row, [](const auto d)
	{
		return std::get<2>(d);
	});
}

// 幾何形状を用いたブロック化多色順序付け
static void GeometicBlockMultiColoring(const Matrix& A)
{
	auto color = std::make_unique<Color[]>(N);
	auto block = std::make_unique<Block[]>(N);

	for(auto i = decltype(n)(0); i < n; i++)
	{
		for(auto j = decltype(n)(0); j < n; j++)
		{
			const auto idx = i*n + j;

			const auto blockI = i / BLOCK_SIZE;
			const auto blockJ = j / BLOCK_SIZE;
			block[idx] = blockI*n / BLOCK_SIZE + blockJ;

			const auto isEvenRow = blockI % 2 == 0;
			const auto isEvenColumn = blockJ % 2 == 0;

			color[idx] = isEvenRow ?
				(isEvenColumn ?
					Color(1) :  // 偶数行の偶数列
					Color(2)) : // 偶数行の奇数列
				(isEvenColumn ?
					Color(3) :  // 奇数業の偶数列
					Color(4));  // 奇数業の奇数列
		}
	}

	// 並び替え後の行番号→元の行番号の変換表
	auto row = std::make_unique<Index[]>(N);
	CreateRow(row.get(), color.get(), block.get());

	OutputResult("幾何的ブロック化多色順序付け", A, row.get(), color.get());
}

// 幾何形状を用いない多色順序付け
static void AlgebraicBlockMultiColoring(const Matrix& A)
{
	constexpr auto INVALID_BLOCK = Block(0);
	auto block = std::make_unique<Block[]>(N);

	constexpr auto MAX_COUNT_PER_BLOCK = BLOCK_SIZE*BLOCK_SIZE; // 1ブロック内の要素数
	constexpr auto MAX_BLOCK_COUNT = N; // 最大は、1ブロック1要素の時
	auto countPerBlock = std::make_unique<std::size_t[]>(MAX_BLOCK_COUNT);
	std::fill_n(countPerBlock.get(), MAX_BLOCK_COUNT, 0);

	// ブロックに割り当て
	auto blockCount = Block(1);
	for(auto i = decltype(N)(0); i < N; i++)
	{
		// まだどのブロックにも入っていなかったら
		if(block[i] == INVALID_BLOCK)
		{
			// 次のブロックに割り当て
			block[i] = blockCount;
			countPerBlock[blockCount]++;

			// まだこのブロックに空きがあれば
			if(countPerBlock[blockCount] < MAX_COUNT_PER_BLOCK)
			{
				auto candidate = std::queue<Index>();
				{
					// 接続している点のうち
					const auto offset = A.index1_data()[i];
					const auto count = A.index1_data()[i + 1] - offset;
					for(auto idx = decltype(count)(0); idx < count; idx++)
					{
						const auto j = A.index2_data()[offset + idx];
						// まだブロックに割り当てられていない点を
						// （自分自身は除く）
						if((i != j) && (block[j] == INVALID_BLOCK))
						{
							// このブロックに入れる候補点にする
							candidate.push(j);
						}
					}
				}

				// ブロックに空きがあれば
				for(auto j = candidate.front(); !candidate.empty() && (countPerBlock[blockCount] < MAX_COUNT_PER_BLOCK); candidate.pop())
				{
					j = candidate.front();

					// 候補点が既に追加済みでなければ
					if(block[j] == INVALID_BLOCK)
					{
						// 候補点をブロックに入れる
						block[j] = blockCount;
						countPerBlock[blockCount]++;

						// まだこのブロックに空きがあれば
						if(countPerBlock[blockCount] < MAX_COUNT_PER_BLOCK)
						{
							// ブロックに入れた点が接続している点のうち
							const auto offset = A.index1_data()[j];
							const auto count = A.index1_data()[j + 1] - offset;
							for(auto idx = decltype(count)(0); idx < count; idx++)
							{
								const auto k = A.index2_data()[offset + idx];
								// まだブロックに割り当てられていない点を
								// （自分自身は除く）
								if((j != k) && (block[k] == INVALID_BLOCK))
								{
									// このブロックに入れる候補点にする
									candidate.push(k);
								}
							}
						}
					}
				}
			}

			blockCount++;
		}
	}
	blockCount--; // ブロック数

	// ブロック間の接続行列を生成
	Matrix M(blockCount, blockCount);
	for(auto i = decltype(N)(0); i < N; i++)
	{
		const auto blockI = block[i] - 1; // ※ブロック番号は1から始まるので

		// この点が接続している点のブロックと接続
		const auto offset = A.index1_data()[i];
		const auto count = A.index1_data()[i + 1] - offset;
		for(auto idx = decltype(count)(0); idx < count; idx++)
		{
			const auto j = A.index2_data()[offset + idx];
			const auto blockJ = block[j] - 1; // ※ブロック番号は1から始まるので

			M(blockI, blockJ) = 1;
		}
	}

	// ブロック間の接続行列に対して代数的（非ブロック化）多色順序付けを実行
	const auto colorBlock = AlgebraicMultiColoringMain(M);

	auto color = std::make_unique<Color[]>(N);
	for(auto i = decltype(N)(0); i < N; i++)
	{
		const auto b = block[i];
		const auto c = colorBlock[b - 1];
		color[i] = c;
	}


	// 並び替え後の行番号→元の行番号の変換表
	auto row = std::make_unique<Index[]>(N);
	CreateRow(row.get(), color.get(), block.get());

	OutputResult("代数的ブロック化多色順序付け", A, row.get(), color.get());
}

int main()
{
	Matrix A(N, N);

#if ENABLE_ROW_DATA
	const double a[N*N] =
	{
		26, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		-1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, -1, 26, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		-1, -1, 0, 0, 26, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		-1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, -1, -1, 0, 0, -1, 26, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, -1, -1, 0, 0, 26, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 26, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 26, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		-1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		-1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 26, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		-1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 26, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		-1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 26, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 26, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 26, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 26, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 26, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 26, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 26, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 26, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 26, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 26, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 26, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 26, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 26, 0, 0, -1, -1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 26, -1, 0, 0, -1, -1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0, -1, -1, -1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 26, 0, 0, -1, -1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 26, -1, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, 26, -1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 26,
	};

	for(auto i = decltype(N)(0); i < N; i++)
	{
		for(auto j = decltype(N)(0); j < N; j++)
		{
			const auto value = a[i*N + j];
			if(value != 0)
			{
				A(i, j) = value;
			}
		}
	}
#else
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

					const auto value = (p == q) ? 26 : -1;
					A(p, q) = value;
				}
			}
		}
	}
#endif

	std::cout << "元行列" << std::endl;
	for(auto i = decltype(N)(0); i < N; i++)
	{
		for(auto j = decltype(N)(0); j < N; j++)
		{
			std::cout << A(i, j) << ", ";
		}
		std::cout << std::endl;
	}

#ifndef ENABLE_ROW_DATA
	GeometicMultiColoring(A);
#endif
	AlgebraicMultiColoring(A);
#ifndef ENABLE_ROW_DATA
	GeometicBlockMultiColoring(A);
#endif
	AlgebraicBlockMultiColoring(A);

	system("pause");
	return 0;
}