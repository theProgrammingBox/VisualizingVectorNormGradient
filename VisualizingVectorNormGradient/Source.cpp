#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;

/*
IMPORTANT LESSONS
1. With Euler, the length of the vector increases very noticeably.
2. With Runge Kutta 4, the length of the vector remains a lot more stable
3. For a learning rate of 0.1, the length of the vector increases by about 0.0001 per frame with runge kutta 4
4. If the initial vector length is very small, with the same learning rate, the length of the vector remains incredibly stable
*/

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
	constexpr float ZEROF = 0.0f;
	constexpr float ONEF = 1.0f;
	constexpr float TWOF = 2.0f;
	constexpr float LEARNING_RATE = 0.1f;
	constexpr float HALF_LEARNING_RATE = LEARNING_RATE * 0.5f;
	constexpr float SIXTH_LEARNING_RATE = LEARNING_RATE * 0.16666666666666666666666666666667f;
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
	
	float invMagsProduct = invSqrt(*sum1 * *sum2);
	
	for (uint32_t j = size; j--;)
		vec1Gradient[j] = (vec2[j] * (*sum1 - vec1[j] * vec1[j]) + vec1[j] * (vec1[j] * vec2[j] - *dot)) * invMagsProduct;
	
	for (uint32_t j = size; j--;)
		vec2Gradient[j] = (vec1[j] * (*sum2 - vec2[j] * vec2[j]) + vec2[j] * (vec2[j] * vec1[j] - *dot)) * invMagsProduct;

	return *dot * invMagsProduct;
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
		vec[0] = 0.001;
		vec[1] = 0;

		mouseVec[0] = -100;
		mouseVec[1] = 0;

		orgin[0] = ScreenWidth() * 0.5f;
		orgin[1] = ScreenHeight() * 0.5f;
		
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		if (GetMouse(0).bHeld)
		{
			mouseVec[0] = GetMouseX() - orgin[0];
			mouseVec[1] = GetMouseY() - orgin[1];
		}
		
		Clear(olc::BLACK);

		DrawLine(orgin[0], orgin[1], orgin[0] + vec[0] * 100000, orgin[1] + vec[1] * 100000, olc::RED);
		DrawLine(orgin[0], orgin[1], orgin[0] + mouseVec[0], orgin[1] + mouseVec[1], olc::GREEN);

		//Runge-Kutta 4th order
		float mouseGrad1[2];
		float vecGrad1[2];
		float mouseGrad2[2];
		float vecGrad2[2];
		float mouseGrad3[2];
		float vecGrad3[2];
		float mouseGrad4[2];
		float vecGrad4[2];
		float mouseVecTemp[2];
		float vecTemp[2];

		cpuNormDot(2, vec, mouseVec, vecGrad1, mouseGrad1);
		memcpy(vecTemp, vec, sizeof(float) * 2);
		memcpy(mouseVecTemp, mouseVec, sizeof(float) * 2);
		cpuSaxpy(2, &GLOBAL::HALF_LEARNING_RATE, mouseGrad1, 1, mouseVecTemp, 1);
		cpuSaxpy(2, &GLOBAL::HALF_LEARNING_RATE, vecGrad1, 1, vecTemp, 1);
		
		cpuNormDot(2, vecTemp, mouseVecTemp, vecGrad2, mouseGrad2);
		memcpy(vecTemp, vec, sizeof(float) * 2);
		memcpy(mouseVecTemp, mouseVec, sizeof(float) * 2);
		cpuSaxpy(2, &GLOBAL::HALF_LEARNING_RATE, mouseGrad2, 1, mouseVecTemp, 1);
		cpuSaxpy(2, &GLOBAL::HALF_LEARNING_RATE, vecGrad2, 1, vecTemp, 1);

		cpuNormDot(2, vecTemp, mouseVecTemp, vecGrad3, mouseGrad3);
		memcpy(vecTemp, vec, sizeof(float) * 2);
		memcpy(mouseVecTemp, mouseVec, sizeof(float) * 2);
		cpuSaxpy(2, &GLOBAL::LEARNING_RATE, mouseGrad3, 1, mouseVecTemp, 1);
		cpuSaxpy(2, &GLOBAL::LEARNING_RATE, vecGrad3, 1, vecTemp, 1);

		cpuNormDot(2, vecTemp, mouseVecTemp, vecGrad4, mouseGrad4);
		cpuSaxpy(2, &GLOBAL::TWOF, mouseGrad2, 1, mouseGrad1, 1);
		cpuSaxpy(2, &GLOBAL::TWOF, vecGrad2, 1, vecGrad1, 1);
		cpuSaxpy(2, &GLOBAL::TWOF, mouseGrad3, 1, mouseGrad1, 1);
		cpuSaxpy(2, &GLOBAL::TWOF, vecGrad3, 1, vecGrad1, 1);
		cpuSaxpy(2, &GLOBAL::ONEF, mouseGrad4, 1, mouseGrad1, 1);
		cpuSaxpy(2, &GLOBAL::ONEF, vecGrad4, 1, vecGrad1, 1);
		cpuSaxpy(2, &GLOBAL::SIXTH_LEARNING_RATE, mouseGrad1, 1, mouseVec, 1);
		cpuSaxpy(2, &GLOBAL::SIXTH_LEARNING_RATE, vecGrad1, 1, vec, 1);
		
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