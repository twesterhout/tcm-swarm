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

#include "nqs.h"
#include "detail/config.hpp"
#include "heisenberg.hpp"
#include "monte_carlo.hpp"
#include "rbm_spin_float.hpp"
#include <cstdio>
#include <complex.h>
#include <memory>

static_assert(
    std::is_same_v<tcm_Index, tcm::Rbm::index_type>, "Oh this is _not_ good!");

namespace {
auto to_span(tcm_Vector& x) -> gsl::span<std::complex<float>>
{
    if (x.size < 0) {
        tcm::global_logger()->critical(
            "Length can't be negative, but got {}.", x.size);
        tcm::throw_with_trace(std::runtime_error{"Negative length."});
    }
    if (x.stride != 1) {
        tcm::global_logger()->critical(
            "Only a vector with stride 1 can be converted to gsl::span, "
            "but got {}.",
            x.stride);
        tcm::throw_with_trace(std::runtime_error{"Invalid stride."});
    }
    return gsl::span{static_cast<std::complex<float>*>(x.data), x.size};
}

[[noreturn]] auto not_implemented()
{
    tcm::global_logger()->critical(
        "Unfortunately, the requested operation is not (yet) supported.",
        TCM_SWARM_ISSUES_LINK);
    tcm::throw_with_trace(std::runtime_error{"Not implemented."});
}

template <class... Ptrs>
auto assert_not_null(char const* name, Ptrs*... pointers)
{
#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wparentheses-equality"
#endif
    if ((... || (pointers == nullptr))) {
        tcm::global_logger()->critical(
            "A nullptr was passed to {}. This is probably a bug. Please, "
            "report it to {}. Terminating now...",
            name, TCM_SWARM_ISSUES_LINK);
        std::terminate();
    }
#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostic pop
#endif
}
} // namespace

#define ASSERT_NOT_NULL(...) assert_not_null(__PRETTY_FUNCTION__, __VA_ARGS__)

// =============================================================================

extern "C" {
TCM_SWARM_SYMBOL_EXPORT
void tcm_Heisenberg_create(tcm_Hamiltonian* const hamiltonian,
    tcm_Index (*edges)[2], tcm_Index const n, int const t1, int const t2,
    float const* cutoff)
{
    tcm::should_not_throw([=]() {
        ASSERT_NOT_NULL(__PRETTY_FUNCTION__, hamiltonian, edges);
        if (n < 0) {
            tcm::global_logger()->critical(
                "Length can't be negative. This is definitely a bug. Please, "
                "report it to {}. Terminating now...",
                TCM_SWARM_ISSUES_LINK);
            std::terminate();
        }
        std::vector<std::array<tcm_Index, 2>> xs(gsl::narrow_cast<std::size_t>(n));
        /*
        for (tcm_Index i = 0; i < n; ++i) {
            tcm::global_logger()->debug("Constructing HH: {}, {} <-> {}",
                i, edges[i][0], edges[i][1]);
            xs[i] = {edges[i][0], edges[i][1]};
        }
        */
        std::memcpy(xs.data(), edges,
            gsl::narrow_cast<std::size_t>(n) * sizeof(tcm_Index[2]));
        hamiltonian->dtype = TCM_SPIN_HEISENBERG;
        hamiltonian->payload =
            cutoff == nullptr
                ? std::make_unique<tcm::Heisenberg>(std::move(xs), std::array{t1, t2})
                      .release()
                : std::make_unique<tcm::Heisenberg>(
                      std::move(xs), *cutoff, std::array{t1, t2})
                      .release();
    });
}

TCM_SWARM_SYMBOL_EXPORT
void tcm_Heisenberg_destroy(tcm_Hamiltonian* const self)
{
    tcm::should_not_throw([=]() {
        ASSERT_NOT_NULL(self);
        if (self->dtype != TCM_SPIN_HEISENBERG) {
            tcm::global_logger()->critical(
                "Not a Heisenberg Hamiltonian was passed to "
                "tcm_Heisenberg_destoy. This is definitely a bug. Please, "
                "report the bug to {}. Terminating now.",
                TCM_SWARM_ISSUES_LINK);
            std::terminate();
        }
        ASSERT_NOT_NULL(self->payload);
        reinterpret_cast<_tcm_Heisenberg*>(self->payload)->~_tcm_Heisenberg();
    });
}
} // extern "C"

// =============================================================================

extern "C" {
TCM_SWARM_SYMBOL_EXPORT
void tcm_Rbm_create(tcm_Rbm* const rbm, tcm_Index const size_visible,
    tcm_Index const size_hidden)
{
    static_assert(sizeof(tcm_Rbm) <= TCM_SWARM_SIZEOF_RBM);
    static_assert(alignof(tcm_Rbm) <= TCM_SWARM_ALIGNOF_RBM);
    tcm::should_not_throw([=]() {
        ASSERT_NOT_NULL(rbm);
        ::new (rbm) _tcm_Rbm{size_visible, size_hidden};
    });
}

TCM_SWARM_SYMBOL_EXPORT
void tcm_Rbm_destroy(tcm_Rbm* const rbm)
{
    tcm::should_not_throw([=]() {
        ASSERT_NOT_NULL(rbm);
        rbm->~_tcm_Rbm();
    });
}

TCM_SWARM_SYMBOL_EXPORT
tcm_Index tcm_Rbm_size(tcm_Rbm const* const rbm)
{
    return tcm::should_not_throw([=]() {
        ASSERT_NOT_NULL(rbm);
        return rbm->size();
    });
}

TCM_SWARM_SYMBOL_EXPORT
tcm_Index tcm_Rbm_size_visible(tcm_Rbm const* const rbm)
{
    return tcm::should_not_throw([=]() {
        ASSERT_NOT_NULL(rbm);
        return rbm->size_visible();
    });
}

TCM_SWARM_SYMBOL_EXPORT
tcm_Index tcm_Rbm_size_hidden(tcm_Rbm const* const rbm)
{
    return tcm::should_not_throw([=]() {
        ASSERT_NOT_NULL(rbm);
        return rbm->size_hidden();
    });
}

TCM_SWARM_SYMBOL_EXPORT
tcm_Index tcm_Rbm_size_weights(tcm_Rbm const* const rbm)
{
    return tcm::should_not_throw([=]() {
        ASSERT_NOT_NULL(rbm);
        return rbm->size_weights();
    });
}

TCM_SWARM_SYMBOL_EXPORT
void tcm_Rbm_get_visible(tcm_Rbm* const rbm, tcm_Vector* const visible)
{
    tcm::should_not_throw([=]() {
        ASSERT_NOT_NULL(rbm, visible);
        *visible = rbm->visible();
    });
}

TCM_SWARM_SYMBOL_EXPORT
void tcm_Rbm_get_hidden(tcm_Rbm* const rbm, tcm_Vector* const hidden)
{
    tcm::should_not_throw([=]() {
        ASSERT_NOT_NULL(rbm, hidden);
        *hidden = rbm->hidden();
    });
}

TCM_SWARM_SYMBOL_EXPORT
void tcm_Rbm_get_weights(tcm_Rbm* const rbm, tcm_Matrix* const weights)
{
    tcm::should_not_throw([=]() {
        ASSERT_NOT_NULL(rbm, weights);
        *weights = rbm->weights();
    });
}
} // extern "C"

// =============================================================================

namespace {
auto to_metropolis_config(tcm_MC_Config const& config) -> tcm::MetropolisConfig
{
    tcm::MetropolisConfig conf;
    conf.steps(tcm::Steps{config.range})
        .threads({config.threads})
        .runs(config.runs)
        .flips(config.flips)
        .restarts(config.restarts);
    if (config.has_magnetisation) {
        conf.magnetisation(config.magnetisation);
    }
    tcm::global_logger()->debug(
        "Configured Monte-Carlo:\n"
        "* Steps: ({}, {}, {})\n"
        "* Threads: ({}, {}, {})\n",
        conf.steps().start(), conf.steps().stop(), conf.steps().step(),
        conf.threads()[0], conf.threads()[1], conf.threads()[2]);
    return conf;
}

constexpr auto no_variance_value() noexcept -> float { return -1.0f; }

} // namespace

extern "C" {
TCM_SWARM_SYMBOL_EXPORT
void tcm_sample_moments(tcm_Rbm const* rbm, tcm_Hamiltonian const* hamiltonian,
    tcm_MC_Config const* config, tcm_Vector* moments, tcm_MC_Stats* stats)
{
    tcm::should_not_throw([=]() {
        ASSERT_NOT_NULL(rbm, hamiltonian, config, moments, stats);
        switch (hamiltonian->dtype) {
        case TCM_SPIN_HEISENBERG: {
            auto const& h =
                *reinterpret_cast<_tcm_Heisenberg const*>(hamiltonian->payload);
            auto [dim, var] = tcm::sample_moments(*rbm, std::cref(h),
                to_metropolis_config(*config), to_span(*moments));
            Expects(!var.has_value() || var.value() >= 0);
            stats->dimension = dim;
            // TODO(twesterhout): This is such an ugly hack...
            stats->variance = std::move(var).value_or(no_variance_value());
            break;
        }
        case TCM_SPIN_ANISOTROPIC_HEISENBERG:
        case TCM_SPIN_DZYALOSHINSKII_MORIYA: not_implemented();
        } // end switch
    });
}

TCM_SWARM_SYMBOL_EXPORT
void tcm_sample_gradients(tcm_Rbm const* rbm,
    tcm_Hamiltonian const* hamiltonian, tcm_MC_Config const* config,
    tcm_Vector* moments, tcm_Vector* force, tcm_Matrix* gradients,
    tcm_MC_Stats* stats)
{
    tcm::should_not_throw([=]() {
        ASSERT_NOT_NULL(
            rbm, hamiltonian, config, moments, force, gradients, stats);
        switch (hamiltonian->dtype) {
        case TCM_SPIN_HEISENBERG: {
            auto const& h =
                *reinterpret_cast<_tcm_Heisenberg const*>(hamiltonian->payload);
            tcm::Gradients<std::complex<float>> grads{*gradients};
            auto [dim, var] = tcm::sample_gradients(*rbm, std::cref(h),
                to_metropolis_config(*config), to_span(*moments),
                to_span(*force), grads);
            Expects(!var.has_value() || var.value() >= 0);
            stats->dimension = dim;
            // TODO(twesterhout): This is such an ugly hack...
            stats->variance = std::move(var).value_or(no_variance_value());
            break;
        }
        case TCM_SPIN_ANISOTROPIC_HEISENBERG:
        case TCM_SPIN_DZYALOSHINSKII_MORIYA: not_implemented();
        } // end switch
    });
}

} // extern "C"

// =============================================================================

extern "C" {

TCM_SWARM_SYMBOL_EXPORT
void tcm_set_log_level(tcm_Level const level)
{
    switch (level) {
    case tcm_Level_trace:
        tcm::global_logger()->set_level(spdlog::level::trace);
        break;
    case tcm_Level_debug:
        tcm::global_logger()->set_level(spdlog::level::debug);
        break;
    case tcm_Level_info:
        tcm::global_logger()->set_level(spdlog::level::info);
        break;
    case tcm_Level_warn:
        tcm::global_logger()->set_level(spdlog::level::warn);
        break;
    case tcm_Level_err:
        tcm::global_logger()->set_level(spdlog::level::err);
        break;
    case tcm_Level_critical:
        tcm::global_logger()->set_level(spdlog::level::critical);
        break;
    case tcm_Level_off:
        tcm::global_logger()->set_level(spdlog::level::off);
        break;
    } // end switch
}

} // extern "C"

#undef ASSERT_NOT_NULL
