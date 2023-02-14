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
	constexpr float LEARNING_RATE = 0.01f;
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

float invSqrt(float number)
{
	long i = 0x5F1FFFF9 - (*(long*)&number >> 1);
	float tmp = *(float*)&i;
	return tmp * 0.703952253f * (2.38924456f - number * tmp * tmp);
}

float cpuNormDot(uint32_t size, float* vec1, float* vec2, float* vec1Gradient, float* vec2Gradient) {
	float sum1[1];
	float sum2[1];
	float dot[1];

	cpuSgemmStridedBatched(
		false, false,
		1, 1, size,
		&GLOBAL::ONEF,
		vec1, 1, 0,
		vec1, size, 0,
		&GLOBAL::ZEROF,
		sum1, 1, 0,
		1);

	cpuSgemmStridedBatched(
		false, false,
		1, 1, size,
		&GLOBAL::ONEF,
		vec2, 1, 0,
		vec2, size, 0,
		&GLOBAL::ZEROF,
		sum2, 1, 0,
		1);

	cpuSgemmStridedBatched(
		false, false,
		1, 1, size,
		&GLOBAL::ONEF,
		vec1, 1, 0,
		vec2, size, 0,
		&GLOBAL::ZEROF,
		dot, 1, 0,
		1);
	
	float denominator = invSqrt(*sum1 * *sum2);
	
	for (uint32_t j = size; j--;)
		vec1Gradient[j] = (vec2[j] * (*sum1 - vec1[j] * vec1[j]) + vec1[j] * (vec1[j] * vec2[j] - *dot)) * denominator;
	
	for (uint32_t j = size; j--;)
		vec2Gradient[j] = (vec1[j] * (*sum2 - vec2[j] * vec2[j]) + vec2[j] * (vec2[j] * vec1[j] - *dot)) * denominator;

	return *dot * denominator;
}

class Visualizer : public olc::PixelGameEngine
{
public:
	float vec[2];
	float mouseVec[2];
	float orgin[2];
	
	Visualizer()
	{
		sAppName = "Visualizing Vector Norm Gradient";
	}

public:
	bool OnUserCreate() override
	{
		cpuGenerateUniform(vec, 2, -60, 60);
		cpuGenerateUniform(mouseVec, 2, -60, 60);

		orgin[0] = ScreenWidth() * 0.5f;
		orgin[1] = ScreenHeight() * 0.5f;
		
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		Clear(olc::BLACK);
		
		if (GetMouse(0).bHeld)
		{
			mouseVec[0] = GetMouseX() - orgin[0];
			mouseVec[1] = GetMouseY() - orgin[1];
		}

		DrawLine(orgin[0], orgin[1], orgin[0] + vec[0] * 10, orgin[1] + vec[1] * 10, olc::RED);
		DrawLine(orgin[0], orgin[1], orgin[0] + mouseVec[0], orgin[1] + mouseVec[1], olc::GREEN);
		
		float mouseGrad[2];
		float vecGrad[2];
		cpuNormDot(2, vec, mouseVec, vecGrad, mouseGrad);
		
		cpuSaxpy(2, &GLOBAL::LEARNING_RATE, vecGrad, 1, vec, 1);
		cpuSaxpy(2, &GLOBAL::LEARNING_RATE, mouseGrad, 1, mouseVec, 1);
		
		float vecMag = sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
		DrawString(10, 10, "vec magnitude: " + std::to_string(vecMag), olc::WHITE, 1);

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