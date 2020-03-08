#pragma once

#if defined (_MSC_VER)  // Visual studio
    #define thread_local __declspec( thread )
#elif defined (__GCC__) // GCC
    #define thread_local __thread
#endif

#include <climits> // CHAR_BIT
#include <random>
#include <chrono>
#include <iterator>
#include <functional>
#include <utility>
#include <type_traits>
//#include <thread>
#include "resources/mingw-std-threads/mingw.thread.h"

/*
 * borowed from: http://mostlymangling.blogspot.com/
 * https://stackoverflow.com/questions/56242943/what-is-the-proper-way-of-seeding-stdmt19937-with-stdchronohigh-resolution
 */
template<typename Out = std::size_t, typename In>
inline typename std::enable_if<sizeof(In) * __CHAR_BIT__ <= 64 && std::numeric_limits<Out>::is_integer && std::numeric_limits<In>::is_integer, Out>::type
sexmex(In v) {
	uint64_t v_uint = static_cast<uint64_t>(v);
	v_uint = (v_uint >> 20) ^ (v_uint >> 37) ^ (v_uint >> 51);
	v_uint *= 0xA54FF53A5F1D36F1ULL; // fractional part of sqrt(7)
	v_uint ^= (v_uint >> 20) ^ (v_uint >> 37) ^ (v_uint >> 51);
	v_uint *= 0x510E527FADE682D1ULL; // fractional part of sqrt(11)
	v_uint ^= (v_uint >> 20) ^ (v_uint >> 37) ^ (v_uint >> 51);
	// Discard the high bits if Out is < 64 bits. This particular hash function
	// has not shown any weaknesses in the lower bits in any widely known test
	// suites yet.
	return static_cast<Out>(v_uint);
}

class PRNG
{
public:
	using res_type = std::uint_least32_t;

	template<typename RandomIt>
	void generate(RandomIt Begin, RandomIt End) const noexcept {
		using seed_t = std::remove_reference_t<decltype(*Begin)>;
		std::random_device rnd{};

		// if no entropy -> add entropy
		if (rnd.entropy() == 0) {
			// returns a time_point with the smallest possible duration i.e. time_point(std::chrono::duration::min())
			constexpr auto min = std::chrono::high_resolution_clock::duration::min();
			std::vector<seed_t> food_for_generator(static_cast<size_t>(std::distance(Begin, End)));

			for (int stiring{ 0 }; stiring < 10; ++stiring) {
				for (auto& food : food_for_generator) {
					// sleep to ensure clock changes each iter
					std::this_thread::sleep_for(min);
					std::this_thread::sleep_for(min);
					auto cc = std::chrono::high_resolution_clock::now().time_since_epoch().count();
					food ^= sexmex<seed_t>(cc);
					food ^= sexmex<seed_t>(rnd());
				}
				stir_buffer(food_for_generator);
			}
			// seed the generator
			for (auto f = food_for_generator.begin(); Begin != End; ++f, ++Begin)
				*Begin = *f;
		}
		else {
			// if entropy -> use random device and make sure rnd n-bits == seed_t n-bits and unbiased via sexmex
			for (; Begin != End; ++Begin) {
				*Begin = sexmex<seed_t>(rnd());
			}
		}
	}

private:
	template<typename SeedType>
	inline void stir_buffer(std::vector<SeedType>& buf) const noexcept {
		for (size_t i{ 0 }; i < buf.size() * 2; ++i) {
			buf[i % buf.size()] += static_cast<SeedType>(sexmex(buf[(i + buf.size() - 1) % buf.size()] + i));
		}
	}
};
//----------------------------------------------------------------------------------
struct shared_generator {
	// we want one instance shared between all instances of uniform_dist per thread
	static thread_local PRNG ss;
	static thread_local std::mt19937 generator;
};

thread_local PRNG shared_generator::ss{};
thread_local std::mt19937 shared_generator::generator(ss); 

//----------------------------------------------------------------------------------
// a distribution template for uniform distributions, both int and real
template<typename T>
class uniform_dist : shared_generator {
public:
	uniform_dist(T inf, T sup) : distribution(inf, sup) {}
	inline T operator()() {
		return distribution(generator); 
	}
private:
	template<typename D>
	using dist_t = std::conditional_t<std::is_integral<D>::value, std::uniform_int_distribution<D>, std::uniform_real_distribution<D>>;
	dist_t<T> distribution;
};
//----------------------------------------------------------------------------------

template<typename T>
class normal_dist : shared_generator {
public:
	normal_dist(T inf, T sup) : distribution(inf, sup) {}
	inline T operator()() { return distribution(generator); }
private:
	template<typename D>
	using dist_t = std::conditional_t<std::is_integral<D>::value, std::binomial_distribution<D>, std::normal_distribution<D>>;
	dist_t<T> distribution;
};
//----------------------------------------------------------------------------------