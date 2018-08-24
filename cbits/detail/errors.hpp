// Copyright Tom Westerhout (c) 2018
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Tom Westerhout nor the names of other
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef TCM_SWARM_DETAIL_ERRORS_HPP
#define TCM_SWARM_DETAIL_ERRORS_HPP

#include "../logging.hpp"
#include "config.hpp"
#include <boost/exception/all.hpp>
#include <boost/stacktrace.hpp>
#include <fmt/ostream.h>
#include <sstream>
#include <type_traits>

TCM_SWARM_BEGIN_NAMESPACE

using errinfo_backtrace = boost::error_info<struct errinfo_backtrace_tag,
    boost::stacktrace::stacktrace>;

template <class E, class = std::enable_if_t<std::is_base_of_v<std::exception,
                       std::remove_cv_t<std::remove_reference_t<E>>>>>
[[noreturn]] auto throw_with_trace(E&& e)
{
    // NOLINTNEXTLINE(hicpp-exception-baseclass)
    throw boost::enable_error_info(std::forward<E>(e))
        << errinfo_backtrace(boost::stacktrace::stacktrace());
}

// clang-format off
template <class Function, class... Args>
TCM_SWARM_FORCEINLINE
decltype(auto) should_not_throw(Function&& func, Args&&... args) noexcept
// clang-format on
{
    try {
        return std::forward<Function>(func)(std::forward<Args>(args)...);
    }
    catch (std::exception const& e) {
        auto const* st = boost::get_error_info<errinfo_backtrace>(e);
        if (st != nullptr) {
            global_logger()->critical(
                "An unrecoverable error occured: an exception was thrown in a "
                "noexcept context.\nDescription: {}\nBacktrace:\n{}\nCalling "
                "terminate now.",
                e.what(), *st);
        }
        else {
            global_logger()->critical(
                "An unrecoverable error occured: an exception was thrown in a "
                "noexcept context.\nDescription: {}\nBacktrace: Not "
                "available.\nCalling terminate now.",
                e.what());
        }
        global_logger()->flush();
        std::terminate();
    }
    catch (...) {
        global_logger()->critical(
            "An unrecoverable error occured: an *unexpected* exception was "
            "thrown in a noexcept context.\nDescription: Not available.\n"
            "Backtrace: Not available.\n Calling terminate now.");
        global_logger()->flush();
        std::terminate();
    }
}

template <class Int>
class negative_size_error
    : public virtual boost::exception
    , public virtual std::invalid_argument {

    static_assert(std::is_integral_v<Int>);

    std::string what_message(char const* arg_name, Int const value)
    {
        std::ostringstream msg;
        msg << "A size cannot be negative, but argument '" << arg_name
            << "' is " << value << ".";
        return msg.str();
    }

  public:
    negative_size_error(char const* arg_name, Int const value)
        : boost::exception{}
        , std::invalid_argument{what_message(arg_name, value)}
    {
    }
};

TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_ERRORS_HPP
