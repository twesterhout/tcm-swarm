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

#pragma once
#include "detail/config.hpp"
#include <gsl/pointers>
#include <spdlog/spdlog.h>
#include <memory>

TCM_SWARM_BEGIN_NAMESPACE

namespace detail {
class BOOST_SYMBOL_VISIBLE _StaticResources {
  public:
    _StaticResources();

    _StaticResources(_StaticResources const&) = delete;
    _StaticResources(_StaticResources&&)      = delete;
    _StaticResources& operator=(_StaticResources const&) = delete;
    _StaticResources& operator=(_StaticResources&&) = delete;

    static auto instance() noexcept -> _StaticResources&;
    auto        logger() noexcept -> gsl::not_null<spdlog::logger*>;

  private:
    std::unique_ptr<spdlog::logger> _logger;
};
} // namespace detail

inline auto global_logger() noexcept -> gsl::not_null<spdlog::logger*>
{
    return detail::_StaticResources::instance().logger();
}

TCM_SWARM_END_NAMESPACE

