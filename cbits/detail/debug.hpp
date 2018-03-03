#ifndef TCM_SWARM_DETAIL_DEBUG_HPP
#define TCM_SWARM_DETAIL_DEBUG_HPP

#include <cstdio>

#include "../detail/config.hpp"

TCM_SWARM_BEGIN_NAMESPACE

namespace detail {

struct Tracer {
    char const* const _name;

    Tracer(char const* name)
        : _name{name}
    {
        std::fprintf(stderr, "<%s>\n", _name);
        std::fflush(stderr);
    }

    ~Tracer()
    {
        std::fprintf(stderr, "</%s>\n", _name);
        std::fflush(stderr);
    }
};

} // namespace detail

#if defined(DO_TRACE)
#define TRACE()                                                      \
    ::TCM_SWARM_NAMESPACE::detail::Tracer _temp_tracer               \
    {                                                                \
        TCM_SWARM_CURRENT_FUNCTION                                   \
    }
#else
#define TRACE()                                                      \
    do {                                                             \
    } while (false)
#endif

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_DEBUG_HPP

