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
static constexpr std::size_t BLOCK_SIZE = 2; // 2x2�Ƀu���b�N��

//#define ENABLE_ROW_DATA

static void OutputResult(const std::string name, const Matrix& A, const Index row[], const Color color[])
{
	// ���̍s�ԍ������ёւ���̍s�ԍ��̕ϊ��\
	auto lut = std::make_unique<Index[]>(N);
	{
		// ���ёւ���̍s�ԍ�����
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

		// �ϊ��\
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

		// �ڑ����Ă���_�Ɏg���Ă���F��􂢏o��
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

// �i��u���b�N���j���F�����t���̍s�ԍ��z��𐶐�
static void CreateRow(Index row[], const Index color[])
{
	// �F�ԍ��̏��������ɕ��ёւ��i�����F�̏ꍇ�͍s�ԍ��̏��������j
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

	// �s�ԍ������̔z��ɕϊ�
	std::transform(pair.get(), pair.get() + N, row, [](const auto p)
	{
		return p.second;
	});
}

// �􉽌`���p�������F�����t��
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
					Color(1) :  // �����s�̋�����
					Color(2)) : // �����s�̊��
				(isEvenColumn ?
					Color(3) :  // ��Ƃ̋�����
					Color(4));  // ��Ƃ̊��
		}
	}

	// ���ёւ���̍s�ԍ������̍s�ԍ��̕ϊ��\
	auto row = std::make_unique<Index[]>(N);
	CreateRow(row.get(), color.get());

	OutputResult("�􉽓I���F�����t��", A, row.get(), color.get());
}

static decltype(auto) AlgebraicMultiColoringMain(const Matrix& A)
{
	const auto size = A.size1();

	auto inNeighbor = std::array<bool, MAX_COLOR_COUNT>();
	constexpr Color INVALID_COLOR = 0;
	auto color = std::vector<Color>(size, INVALID_COLOR);
	color[0] = Color(1); // �ŏ��̓_�͐F1

	for(auto i = decltype(size)(1); i < size; i++)
	{
		const auto offset = A.index1_data()[i];
		const auto count = A.index1_data()[i + 1] - offset;

		// �ڑ����Ă���_�Ɏg���Ă���F��􂢏o��
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

		// �g���Ă��Ȃ��F�̂����ŏ��l�������̐F�ɂ���
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

// �􉽌`���p���Ȃ����F�����t��
static void AlgebraicMultiColoring(const Matrix& A)
{
	const auto&& color = AlgebraicMultiColoringMain(A);

	// ���ёւ���̍s�ԍ������̍s�ԍ��̕ϊ��\
	auto row = std::make_unique<Index[]>(N);
	CreateRow(row.get(), color.data());

	OutputResult("�㐔�I���F�����t��", A, row.get(), color.data());
}

// �i��u���b�N���j���F�����t���̍s�ԍ��z��𐶐�
static void CreateRow(Index row[], const Index color[], const Block block[])
{
	// �F�ԍ��̏��������ɕ��ёւ�
	// �E�����F�̏ꍇ�̓u���b�N�ԍ��̏�������
	// �E�����u���b�N�Ȃ�s�ԍ��̏�������
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

	// �s�ԍ������̔z��ɕϊ�
	std::transform(data.get(), data.get() + N, row, [](const auto d)
	{
		return std::get<2>(d);
	});
}

// �􉽌`���p�����u���b�N�����F�����t��
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
					Color(1) :  // �����s�̋�����
					Color(2)) : // �����s�̊��
				(isEvenColumn ?
					Color(3) :  // ��Ƃ̋�����
					Color(4));  // ��Ƃ̊��
		}
	}

	// ���ёւ���̍s�ԍ������̍s�ԍ��̕ϊ��\
	auto row = std::make_unique<Index[]>(N);
	CreateRow(row.get(), color.get(), block.get());

	OutputResult("�􉽓I�u���b�N�����F�����t��", A, row.get(), color.get());
}

// �􉽌`���p���Ȃ����F�����t��
static void AlgebraicBlockMultiColoring(const Matrix& A)
{
	constexpr auto INVALID_BLOCK = Block(0);
	auto block = std::make_unique<Block[]>(N);

	constexpr auto MAX_COUNT_PER_BLOCK = BLOCK_SIZE*BLOCK_SIZE; // 1�u���b�N���̗v�f��
	constexpr auto MAX_BLOCK_COUNT = N; // �ő�́A1�u���b�N1�v�f�̎�
	auto countPerBlock = std::make_unique<std::size_t[]>(MAX_BLOCK_COUNT);
	std::fill_n(countPerBlock.get(), MAX_BLOCK_COUNT, 0);

	// �u���b�N�Ɋ��蓖��
	auto blockCount = Block(1);
	for(auto i = decltype(N)(0); i < N; i++)
	{
		// �܂��ǂ̃u���b�N�ɂ������Ă��Ȃ�������
		if(block[i] == INVALID_BLOCK)
		{
			// ���̃u���b�N�Ɋ��蓖��
			block[i] = blockCount;
			countPerBlock[blockCount]++;

			// �܂����̃u���b�N�ɋ󂫂������
			if(countPerBlock[blockCount] < MAX_COUNT_PER_BLOCK)
			{
				auto candidate = std::queue<Index>();
				{
					// �ڑ����Ă���_�̂���
					const auto offset = A.index1_data()[i];
					const auto count = A.index1_data()[i + 1] - offset;
					for(auto idx = decltype(count)(0); idx < count; idx++)
					{
						const auto j = A.index2_data()[offset + idx];
						// �܂��u���b�N�Ɋ��蓖�Ă��Ă��Ȃ��_��
						// �i�������g�͏����j
						if((i != j) && (block[j] == INVALID_BLOCK))
						{
							// ���̃u���b�N�ɓ������_�ɂ���
							candidate.push(j);
						}
					}
				}

				// �u���b�N�ɋ󂫂������
				for(auto j = candidate.front(); !candidate.empty() && (countPerBlock[blockCount] < MAX_COUNT_PER_BLOCK); candidate.pop())
				{
					j = candidate.front();

					// ���_�����ɒǉ��ς݂łȂ����
					if(block[j] == INVALID_BLOCK)
					{
						// ���_���u���b�N�ɓ����
						block[j] = blockCount;
						countPerBlock[blockCount]++;

						// �܂����̃u���b�N�ɋ󂫂������
						if(countPerBlock[blockCount] < MAX_COUNT_PER_BLOCK)
						{
							// �u���b�N�ɓ��ꂽ�_���ڑ����Ă���_�̂���
							const auto offset = A.index1_data()[j];
							const auto count = A.index1_data()[j + 1] - offset;
							for(auto idx = decltype(count)(0); idx < count; idx++)
							{
								const auto k = A.index2_data()[offset + idx];
								// �܂��u���b�N�Ɋ��蓖�Ă��Ă��Ȃ��_��
								// �i�������g�͏����j
								if((j != k) && (block[k] == INVALID_BLOCK))
								{
									// ���̃u���b�N�ɓ������_�ɂ���
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
	blockCount--; // �u���b�N��

	// �u���b�N�Ԃ̐ڑ��s��𐶐�
	Matrix M(blockCount, blockCount);
	for(auto i = decltype(N)(0); i < N; i++)
	{
		const auto blockI = block[i] - 1; // ���u���b�N�ԍ���1����n�܂�̂�

		// ���̓_���ڑ����Ă���_�̃u���b�N�Ɛڑ�
		const auto offset = A.index1_data()[i];
		const auto count = A.index1_data()[i + 1] - offset;
		for(auto idx = decltype(count)(0); idx < count; idx++)
		{
			const auto j = A.index2_data()[offset + idx];
			const auto blockJ = block[j] - 1; // ���u���b�N�ԍ���1����n�܂�̂�

			M(blockI, blockJ) = 1;
		}
	}

	// �u���b�N�Ԃ̐ڑ��s��ɑ΂��đ㐔�I�i��u���b�N���j���F�����t�������s
	const auto colorBlock = AlgebraicMultiColoringMain(M);

	auto color = std::make_unique<Color[]>(N);
	for(auto i = decltype(N)(0); i < N; i++)
	{
		const auto b = block[i];
		const auto c = colorBlock[b - 1];
		color[i] = c;
	}


	// ���ёւ���̍s�ԍ������̍s�ԍ��̕ϊ��\
	auto row = std::make_unique<Index[]>(N);
	CreateRow(row.get(), color.get(), block.get());

	OutputResult("�㐔�I�u���b�N�����F�����t��", A, row.get(), color.get());
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

	std::cout << "���s��" << std::endl;
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