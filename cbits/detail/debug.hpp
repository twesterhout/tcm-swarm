#ifndef TCM_SWARM_DETAIL_DEBUG_HPP
#define TCM_SWARM_DETAIL_DEBUG_HPP

#include <cstdio>

#include "../detail/config.hpp"

TCM_SWARM_BEGIN_NAMESPACE

namespace detail {

struct Tracer {
    char const* const _at_enter;
    char const* const _at_exit;

    Tracer(char const* at_enter, char const* at_exit)
        : _at_enter{at_enter}, _at_exit{at_exit}
    {
        std::fprintf(stderr, "%s\n", _at_enter);
        std::fflush(stderr);
    }

    ~Tracer()
    {
        std::fprintf(stderr, "%s\n", _at_exit);
        std::fflush(stderr);
    }
};

#if defined(DO_TRACE)
#define TRACE()                                                      \
    ::TCM_SWARM_NAMESPACE::detail::Tracer _temp_tracer               \
    {                                                                \
        "<" TCM_SWARM_CURRENT_FUNCTION ">",                          \
            "</" TCM_SWARM_CURRENT_FUNCTION ">"                      \
    }
#else
#define TRACE()                                                      \
    do {                                                             \
    } while (false)
#endif

} // namespace detail

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_DEBUG_HPP

