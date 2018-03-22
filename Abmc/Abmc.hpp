#ifndef ABMC_HPP_INCLUDED
#define ABMC_HPP_INCLUDED
#include <iostream>
#include <algorithm>
#include <queue>
#include <set>
#include "boost/format.hpp"

#include "common.hpp"
#include "SymGS.hpp"

static constexpr std::size_t MAX_COLOR_COUNT = 27;

static void OutputResult(const std::string name, const Matrix& A, const Index row[])
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
}

// （非ブロック化）多色順序付けの行番号配列を生成
static void CreateRow(Index row[], Index offset[], const Index color[])
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

	// 各色の先頭番号を計算
	{
		offset[0] = 0;
		for(auto i = decltype(N)(1); i < N; i++)
		{
			const auto prevColor = color[row[i - 1]] - 1;
			const auto thisColor = color[row[i]] - 1;

			if(prevColor != thisColor)
			{
				offset[thisColor] = i;
			}
		}
		const auto lastColor = color[N - 1];
		offset[lastColor] = N;
	}
}

// 幾何形状を用いた多色順序付け
static void GeometicMultiColoring(const Matrix& A, const Vector& b, const Vector& expect)
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
	const auto colorCount = Color(4);

	auto row = std::make_unique<Index[]>(N); // 並び替え後の行番号→元の行番号の変換表
	auto offset = std::make_unique<Index[]>(colorCount + 1); // 各色の開始番号
	CreateRow(row.get(), offset.get(), color.get());

	OutputResult("幾何的多色順序付け", A, row.get());
	GaussSeidel(A, b, expect, row.get(), offset.get(), colorCount);
	SymmetryGaussSeidel(A, b, expect, row.get(), offset.get(), colorCount);
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
static void AlgebraicMultiColoring(const Matrix& A, const Vector& b, const Vector& expect)
{
	const auto&& color = AlgebraicMultiColoringMain(A);
	const auto colorCount = color.size();

	auto row = std::make_unique<Index[]>(N); // 並び替え後の行番号→元の行番号の変換表
	auto offset = std::make_unique<Index[]>(colorCount + 1); // 各色の開始番号
	CreateRow(row.get(), offset.get(), color.data());

	OutputResult("代数的多色順序付け", A, row.get());
	GaussSeidel(A, b, expect, row.get(), offset.get(), colorCount);
	SymmetryGaussSeidel(A, b, expect, row.get(), offset.get(), colorCount);
}

// ブロック化多色順序付けの行番号配列を生成
static void CreateRow(Index row[], Index offset[], Block blockOffset[], const Index color[], const Block block[])
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

	// 各ブロックの先頭番号
	auto blockCount = Block(1);
	{
		offset[0] = 0;
		for(auto i = decltype(N)(1); i < N; i++)
		{
			const auto prevBlock = std::get<1>(data[i - 1]);
			const auto thisBlock = std::get<1>(data[i]);
			if(prevBlock != thisBlock)
			{
				offset[blockCount] = i;
				blockCount++;
			}
		}
		offset[blockCount] = N;
	}

	// 各色のブロック番号
	{
		blockOffset[0] = 0;
		auto colorCount = Color(1);
		for(auto b = Block(1); b < blockCount; b++)
		{
			const auto prevI = offset[b - 1];
			const auto thisI = offset[b];
			const auto prevBlockFirst = row[prevI];
			const auto thisBlockFirst = row[thisI];
			const auto prevColor = color[prevBlockFirst] - 1;
			const auto thisColor = color[thisBlockFirst] - 1;

			if(prevColor != thisColor)
			{
				blockOffset[thisColor] = b;
			}
			colorCount = std::max(colorCount, thisColor);
		}
		colorCount++;
		blockOffset[colorCount] = blockCount;
	}
}

// 幾何形状を用いたブロック化多色順序付け
template<std::size_t BLOCK_SIZE>
static void GeometicBlockMultiColoring(const Matrix& A, const Vector& b, const Vector& expect)
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
			block[idx] = blockI*n / BLOCK_SIZE + blockJ + 1;

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
	const auto blockCountPerLine = static_cast<Block>(std::ceil(static_cast<double>(n) / BLOCK_SIZE));
	const auto blockCount = blockCountPerLine*blockCountPerLine;
	const auto colorCount = 4;

	// 並び替え後の行番号→元の行番号の変換表
	auto row = std::make_unique<Index[]>(N);
	auto offset = std::make_unique<Index[]>(blockCount + 1);
	auto blockOffset = std::make_unique<Block[]>(colorCount + 1);
	CreateRow(row.get(), offset.get(), blockOffset.get(), color.get(), block.get());

	OutputResult((boost::format("幾何的ブロック化多色順序付け(%1%x%1%)") % BLOCK_SIZE).str(), A, row.get());
	GaussSeidel(A, b, expect, row.get(), blockOffset.get(), offset.get(), colorCount);
	SymmetryGaussSeidel(A, b, expect, row.get(), blockOffset.get(), offset.get(), colorCount);
}

// 幾何形状を用いない多色順序付け
template<std::size_t BLOCK_SIZE>
static void AlgebraicBlockMultiColoring(const Matrix& A, const Vector& b, const Vector& expect)
{
	constexpr auto INVALID_BLOCK = Block(0);
	auto block = std::make_unique<Block[]>(N);
	std::fill_n(block.get(), N, INVALID_BLOCK);

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
				for(;!candidate.empty() && (countPerBlock[blockCount] < MAX_COUNT_PER_BLOCK); candidate.pop())
				{
					const auto j = candidate.front();

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
	const auto colorCount = *std::max_element(colorBlock.begin(), colorBlock.end());

	// 並び替え後の行番号→元の行番号の変換表
	auto row = std::make_unique<Index[]>(N);
	auto offset = std::make_unique<Index[]>(blockCount + 1);
	auto blockOffset = std::make_unique<Block[]>(colorCount + 1);
	CreateRow(row.get(), offset.get(), blockOffset.get(), color.get(), block.get());

	OutputResult((boost::format("代数的ブロック化多色順序付け(%1%x%1%)") % BLOCK_SIZE).str(), A, row.get());
	GaussSeidel(A, b, expect, row.get(), blockOffset.get(), offset.get(), colorCount);
	SymmetryGaussSeidel(A, b, expect, row.get(), blockOffset.get(), offset.get(), colorCount);
}

static void CuthillMckee(const Matrix& A, const Vector& b, const Vector& expect)
{
	constexpr auto INVALID_LEVEL = Level(0);
	auto level = std::make_unique<Level[]>(N);
	std::fill(level.get(), level.get() + N, INVALID_LEVEL);
	auto degreeIndex = std::make_unique<std::pair<Index, Index>[]>(N); // degree, index
	auto degree = std::make_unique<Index[]>(N);

	// 次数を求める
	for (auto i = decltype(N)(0); i < N; i++)
	{
		const auto offset = A.index1_data()[i];
		const auto deg = A.index1_data()[i+1] - offset; // count
		degreeIndex[i] = std::make_pair(deg, i);
		degree[i] = deg;
	}

	// 1. 最小次数を探す O(N)
	std::sort(degreeIndex.get(), degreeIndex.get() + N);
	auto maxLevel = Level(0);

	// 点群が別れているときに、再度最小次数を探して幅優先探索をするためのループ O(N)
	while (true)
	{
		// 探索済みではない次数が最小となるindexの探索
		auto minDegreeIndexItr = std::find_if(degreeIndex.get(), degreeIndex.get() + N, [&level = static_cast<const std::unique_ptr<Level[]>&>(level), INVALID_LEVEL](const auto& degIndex)
		{
			return level[degIndex.second] == INVALID_LEVEL;
		});

		// すべての点を探索し終えたら終了
		if (minDegreeIndexItr == degreeIndex.get() + N) break;

		const auto minDegreeIndex = minDegreeIndexItr->second;

		// 最小次数の点を初期探索点とする(最初に呼ばれたときはレベル1スタート)
		std::queue<Level> que;
		que.push(minDegreeIndex);
		level[minDegreeIndex] = maxLevel + 1;
		maxLevel = level[minDegreeIndex];

		//幅優先探索
		std::set<Index> adjacentPoint; //同じレベル内の隣接点をメモ
		Level prevLevel = maxLevel; // 一つ前に代入したレベル
		for (; !que.empty(); que.pop())
		{
			const auto i = que.front();
			const auto levelI = level[i];
			const auto offset = A.index1_data()[i];
			const auto count = A.index1_data()[i+1] - offset; // count
			maxLevel = levelI + 1;
			// 新しいレベルの探索になるため、隣接点をクリア
			if (prevLevel < maxLevel)
			{
				adjacentPoint.clear();
			}
			for (auto idx = decltype(count)(0); idx < count; idx++)
			{
				const auto j = A.index2_data()[offset + idx];
				// 隣接点の levelが未割り当て かつ 同じレベルのすでに割り当てられる点に隣接していない
				if (level[j] == INVALID_LEVEL && adjacentPoint.find(j) == adjacentPoint.end()) {
					level[j] = maxLevel;
					prevLevel = maxLevel;
					que.push(j);

					// 隣接点の隣接点用のoffsetとcount
					const auto offsetJ = A.index1_data()[j];
					const auto countJ = A.index1_data()[j+1] - offsetJ;
					for (int jdx = decltype(countJ)(0); jdx < countJ; jdx++)
					{
						// jj: 隣接点の隣接点
						const auto jj = A.index2_data()[offsetJ + jdx];
						// jjが未探索(探索済みのIndexを登録する意味はない)
						if (level[jj] == INVALID_LEVEL) {
							adjacentPoint.insert(jj);
						}
					}
				}
			}
		}
		maxLevel--; //最後のqueが読まれたタイミングでmaxLevelが1大きくなるためここで減らす
	}

	auto row = std::make_unique<Index[]>(N); // 並び替え後の行番号→元の行番号の変換表
	auto offset = std::make_unique<Index[]>(maxLevel+1); // 各レベルの開始番号
	// rowとoffsetの作成
	{
		// レベルの小さい順に並び替え
		// ・同じレベルの場合は次数の小さい順
		// ・同じ次数なら行番号の小さい順
		auto data = std::make_unique<std::tuple<Level, Index, Index>[]>(N); // Level, degree, index
		for (auto i = decltype(N)(0); i < N; i++)
		{
			data[i] = std::make_tuple(level[i], degree[i], i);
		}
		std::sort(data.get(), data.get() + N, [](auto left, auto right)
		{
			{
				const auto leftLevel = std::get<0>(left); const auto rightLevel = std::get<0>(right);
				if (leftLevel != rightLevel)
				{
					return leftLevel < rightLevel;
				}
			}
			{
				const auto leftDegree = std::get<1>(left); const auto rightDegree = std::get<1>(right);
				if (leftDegree != rightDegree)
				{
					return leftDegree < rightDegree;
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
		std::transform(data.get(), data.get() + N, row.get(), [](const auto d)
		{
			return std::get<2>(d);
		});

		// 各Levelの先頭番号
		auto levelCount = Level(1);
		{
			offset[0] = 0;
			for (auto i = decltype(N)(1); i < N; i++)
			{
				const auto prevLevel = std::get<0>(data[i - 1]);
				const auto thisLevel = std::get<0>(data[i]);
				if (prevLevel != thisLevel)
				{
					offset[levelCount] = i;
					levelCount++;
				}
			}
			offset[levelCount] = N;
		}
	}
	GaussSeidelForCuthillMckee(A,b,expect, row.get(), offset.get(), maxLevel);
	SymmetryGaussSeidelForCuthillMckee(A,b,expect, row.get(), offset.get(), maxLevel);
}

#endif
