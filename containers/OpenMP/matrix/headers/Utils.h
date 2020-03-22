#include <chrono>

using sec_t   = std::chrono::seconds;
using milli_t = std::chrono::milliseconds;
using micro_t = std::chrono::microseconds;
using nano_t  = std::chrono::nanoseconds;

template<typename duration_t>
class Timer{
	using clock_t = std::chrono::system_clock;

public:
	Timer() : _start{ getNow() }{ }
	
	uint64_t getNow() {
		return std::chrono::duration_cast<duration_t>(clock_t::now().time_since_epoch()).count();
	}
	void reset(){ _start = getNow(); }
	uint64_t elapsed() { return getNow() - _start; }
	
private:
	uint64_t _start;
};
