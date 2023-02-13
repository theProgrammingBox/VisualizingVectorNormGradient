#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;

class Random
{
public:
	Random(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, 4, seed);
		state[1] = Hash((uint8_t*)&seed, 4, state[0]);
	}

	static uint32_t MakeSeed(uint32_t seed = 0)	// make seed from time and seed
	{
		uint32_t result = seed;
		result = Hash((uint8_t*)&result, 4, nanosecond());
		result = Hash((uint8_t*)&result, 4, microsecond());
		return result;
	}

	void Seed(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, 4, seed);
		state[1] = Hash((uint8_t*)&seed, 4, state[0]);
	}

	uint32_t Ruint32()	// XORSHIFT128+
	{
		uint64_t a = state[0];
		uint64_t b = state[1];
		state[0] = b;
		a ^= a << 23;
		state[1] = a ^ b ^ (a >> 18) ^ (b >> 5);
		return uint32_t((state[1] + b) >> 16);
	}

	float Rfloat(float min = 0, float max = 1) { return min + (max - min) * Ruint32() * 2.3283064371e-10; }

	static uint32_t Hash(const uint8_t* key, size_t len, uint32_t seed = 0)	// MurmurHash3
	{
		uint32_t h = seed;
		uint32_t k;
		for (size_t i = len >> 2; i; i--) {
			memcpy(&k, key, 4);
			key += 4;
			h ^= murmur_32_scramble(k);
			h = (h << 13) | (h >> 19);
			h = h * 5 + 0xe6546b64;
		}
		k = 0;
		for (size_t i = len & 3; i; i--) {
			k <<= 8;
			k |= key[i - 1];
		}
		h ^= murmur_32_scramble(k);
		h ^= len;
		h ^= h >> 16;
		h *= 0x85ebca6b;
		h ^= h >> 13;
		h *= 0xc2b2ae35;
		h ^= h >> 16;
		return h;
	}

private:
	uint64_t state[2];

	static uint32_t murmur_32_scramble(uint32_t k) {
		k *= 0xcc9e2d51;
		k = (k << 15) | (k >> 17);
		k *= 0x1b873593;
		return k;
	}

	static uint32_t nanosecond() { return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count(); }
	static uint32_t microsecond() { return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count(); }
};

namespace GLOBAL
{
	Random random(Random::MakeSeed(0));
	constexpr float ONEF = 1.0f;
	constexpr float ZEROF = 0.0f;
	constexpr float LEARNING_RATE = 0.1f;
}

void cpuGenerateUniform(float* matrix, uint32_t size, float min = 0, float max = 1)
{
	for (uint32_t counter = size; counter--;)
		matrix[counter] = GLOBAL::random.Rfloat(min, max);
}

void cpuSgemmStridedBatched(
	bool transB, bool transA,
	int CCols, int CRows, int AColsBRows,
	const float* alpha,
	float* B, int ColsB, int SizeB,
	float* A, int ColsA, int SizeA,
	const float* beta,
	float* C, int ColsC, int SizeC,
	int batchCount)
{
	for (int b = batchCount; b--;)
	{
		for (int m = CCols; m--;)
			for (int n = CRows; n--;)
			{
				float sum = 0;
				for (int k = AColsBRows; k--;)
					sum += (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]);
				C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

void cpuSaxpy(int N, const float* alpha, const float* X, int incX, float* Y, int incY)
{
	for (int i = N; i--;)
		Y[i * incY] += *alpha * X[i * incX];
}

float fastInvSqrt(float number)
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y = number;
	i = *(long*)&y;
	i = 0x5f3759df - (i >> 1);
	y = *(float*)&i;
	y = y * (threehalfs - (x2 * y * y));

	return y;
}

class Visualizer : public olc::PixelGameEngine
{
public:
	float vec[2];
	float orgin[2];
	
	Visualizer()
	{
		sAppName = "Visualizing Vector Norm Gradient";
	}

public:
	bool OnUserCreate() override
	{
		cpuGenerateUniform(vec, 2, -100, 100);

		orgin[0] = ScreenWidth() * 0.5f;
		orgin[1] = ScreenHeight() * 0.5f;
		
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		Clear(olc::BLACK);
		
		// normalize mouse vector
		float mouseVec[2];
		mouseVec[0] = GetMouseX() - orgin[0];
		mouseVec[1] = GetMouseY() - orgin[1];
		float invMag = fastInvSqrt(mouseVec[0] * mouseVec[0] + mouseVec[1] * mouseVec[1]);
		mouseVec[0] *= invMag;
		mouseVec[1] *= invMag;

		// normalize vector
		invMag = fastInvSqrt(vec[0] * vec[0] + vec[1] * vec[1]);
		float dx[2];
		dx[0] = vec[0] * invMag;
		dx[1] = vec[1] * invMag;

		DrawLine(orgin[0], orgin[1], orgin[0] + vec[0], orgin[1] + vec[1], olc::RED);
		DrawLine(orgin[0], orgin[1], orgin[0] + mouseVec[0] * 100, orgin[1] + mouseVec[1] * 100, olc::GREEN);
		DrawLine(orgin[0], orgin[1], orgin[0] + dx[0] * 100, orgin[1] + dx[1] * 100, olc::BLUE);

		// calculate dot product
		float dot[1];
		cpuSgemmStridedBatched(
			false, false,
			1, 1, 2,
			&GLOBAL::ONEF,
			mouseVec, 1, 0,
			dx, 2, 0,
			&GLOBAL::ZEROF,
			dot, 1, 0,
			1);

		// calculate gradient
		float err[1];
		err[0] = 1.0f;
		float grad[2];
		cpuSgemmStridedBatched(
			true, false,
			2, 1, 1,
			&GLOBAL::ONEF,
			mouseVec, 1, 0,
			err, 1, 0,
			&GLOBAL::ZEROF,
			grad, 2, 0,
			1);

		//DrawString(0, 0, "Gradient: " + std::to_string(grad[0]) + ", " + std::to_string(grad[1]));

		float vecGrad[2];
		float total = vec[0] * vec[0] + vec[1] * vec[1];
		vecGrad[0] = ((grad[0] * vec[1] * vec[1]) - (grad[1] * vec[0] * vec[1])) / std::pow(total, 1.5f);
		vecGrad[0] = ((grad[1] * vec[0] * vec[0]) - (grad[0] * vec[1] * vec[0])) / std::pow(total, 1.5f);
		
		// apply gradient
		cpuSaxpy(2, &GLOBAL::LEARNING_RATE, grad, 1, vec, 1);

		return true;
	}
};

int main()
{
	Visualizer visualizer;
	if (visualizer.Construct(960, 540, 1, 1))
		visualizer.Start();
	return 0;
}