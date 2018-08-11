
#include <cstdio>
#include <memory>

#include "nqs.h"
#include "rbm_spin.hpp"
#include "heisenberg.hpp"
#include "monte_carlo.hpp"


struct _tcm_cRbm : public tcm::RbmBase<std::complex<float>> {};

namespace {

#define DEFINE_TO_CXX_TYPE_FN(c_type, cxx_type)                                \
    auto* to_cxx_type(c_type* x) noexcept                                      \
    {                                                                          \
        return reinterpret_cast<cxx_type*>(x);                                 \
    }

// clang-format off
DEFINE_TO_CXX_TYPE_FN(tcm_cRbm, tcm::RbmBase<std::complex<float>>);
DEFINE_TO_CXX_TYPE_FN(tcm_cRbm const, tcm::RbmBase<std::complex<float>> const);
// clang-format on

#undef DEFINE_TO_CXX_TYPE_FN

template <class Rbm, class... Args>
int _inplace_construc_rbm(Rbm* const p, Args&&... args) noexcept
{
    try {
        new (p) Rbm(std::forward<Args>(args)...);
        return 0;
    }
    catch (std::exception const& e) {
        std::cerr << "[-] Error: " << e.what() << '\n';
        auto const* st = boost::get_error_info<tcm::traced>(e);
        if (st != nullptr) {
            std::cerr << "Backtrace:\n" << *st << '\n';
        }
        std::cerr << std::flush;
        std::exit(1);
        TCM_SWARM_UNREACHABLE;
    }
    catch (...) {
        std::cerr << "[-] PANIC! " << TCM_SWARM_CURRENT_FUNCTION
                  << ": Unexpected exception!\n"
                  << std::flush;
        std::exit(1);
        TCM_SWARM_UNREACHABLE;
    }
}

template <class Function, class... Args>
decltype(auto) _should_not_throw(Function&& func, Args&&... args) noexcept
{
    try {
        return std::invoke(
            std::forward<Function>(func), std::forward<Args>(args)...);
    }
    catch (std::exception const& e) {
        std::cerr << "[-] Error: " << e.what() << '\n';
        auto const* st = boost::get_error_info<tcm::traced>(e);
        if (st != nullptr) {
            std::cerr << "Backtrace:\n" << *st << '\n';
        }
        std::cerr << std::flush;
        std::exit(1);
        TCM_SWARM_UNREACHABLE;
    }
}
} // namespace

extern "C"
tcm_cRbm* tcm_cRbm_Create(int const size_visible, int const size_hidden)
{
    return _should_not_throw(
        [](auto const n, auto const m) {
            return reinterpret_cast<tcm_cRbm*>(
                std::make_unique<tcm::RbmBase<std::complex<float>>>(n, m)
                    .release());
        },
        size_visible, size_hidden);
}

extern "C"
void tcm_cRbm_Destroy(tcm_cRbm* const p)
{
    using Rbm = tcm::RbmBase<std::complex<float>>;
    to_cxx_type(p)->~Rbm();
}

extern "C"
int tcm_cRbm_Size(tcm_cRbm const* const rbm)
{
    return to_cxx_type(rbm)->size();
}

extern "C"
int tcm_cRbm_Size_visible(tcm_cRbm const* const rbm)
{
    return to_cxx_type(rbm)->size_visible();
}

extern "C"
int tcm_cRbm_Size_hidden(tcm_cRbm const* const rbm)
{
    return to_cxx_type(rbm)->size_hidden();
}

extern "C"
void tcm_cRbm_Get_visible(tcm_cRbm* const rbm, tcm_cTensor1* const visible)
{
    auto a           = to_cxx_type(rbm)->visible();
    visible->data    = reinterpret_cast<tcm_Complex8*>(a.data());
    visible->extents[0] = a.size();
    // std::cout << "tcm_cRbm_Get_visible: " << a.size() << '\n';
}

extern "C"
void tcm_cRbm_Get_hidden(tcm_cRbm* const rbm, tcm_cTensor1* const hidden)
{
    auto b          = to_cxx_type(rbm)->hidden();
    hidden->data    = reinterpret_cast<tcm_Complex8*>(b.data());
    hidden->extents[0] = b.size();
}

extern "C"
void tcm_cRbm_Get_weights(tcm_cRbm* const rbm, tcm_cTensor2* const weights)
{
    auto w           = to_cxx_type(rbm)->weights();
    weights->data    = reinterpret_cast<tcm_Complex8*>(w.data());
    weights->extents[0] = to_cxx_type(rbm)->size_visible();
    weights->extents[1] = to_cxx_type(rbm)->size_hidden();
}

void tcm_cRbm_Sample(
    tcm_cRbm const* const rbm,
    tcm_Hamiltonian const hamiltonian,
    int const number_runs,
    tcm_Range const* const steps,
    int const* const magnetisation,
    tcm_cEstimate* energy,
    tcm_cTensor1* force,
    tcm_cTensor2* derivatives)
{
    std::cerr << "Not implemented #3!\n" << std::flush;
    std::exit(1);
}


extern "C"
void tcm_cRbm_Sample_Moments(tcm_cRbm const* const rbm,
    tcm_Hamiltonian const hamiltonian, int const number_runs,
    tcm_Range const* const steps, int const* const magnetisation,
    tcm_cTensor1* moments)
{
    auto const& state = *to_cxx_type(rbm);

    if (moments->extents[0] != 4) {
        std::cerr << "Not implemented #1!\n" << std::flush;
        std::exit(1);
    }

    switch (hamiltonian) {
    case HEISENBERG_1D_OPEN:
        std::cerr << "Not implemented #2!\n" << std::flush;
        std::exit(1);
    case HEISENBERG_1D_PERIODIC: {
        auto h = tcm::Heisenberg<float, 1, true>{5.0};
        auto s = std::make_tuple(steps->start, steps->step, steps->stop);
        auto m = magnetisation == nullptr ? std::nullopt
                                          : std::optional{*magnetisation};
        auto result = tcm::sample_moments<4>(
            state, h, number_runs, std::move(s), 2, m, 5);
        std::copy(std::begin(result), std::end(result),
            reinterpret_cast<std::complex<float>*>(moments->data));
    }
    } // end switch
}

/*
extern "C" void tcm_cRBM_Set_visible(tcm_cRBM* const rbm,
    tcm_Complex8 const* const visible, tcm_size_type const size)
{
    TRACE();
    _should_not_throw(
        [](auto& x, auto const* w, auto const n) {
            x.unsafe_update_visible(gsl::span{w, n});
        },
        *to_cxx_type(rbm),
        static_cast<std::complex<float> const*>(visible), size);
}

extern "C" void tcm_cRBM_Set_hidden(tcm_cRBM* const rbm,
    tcm_Complex8 const* const hidden, tcm_size_type const size)
{
    TRACE();
    _should_not_throw(
        [](auto& x, auto const* w, auto const n) {
            x.unsafe_update_visible(gsl::span{w, n});
        },
        *to_cxx_type(rbm),
        static_cast<std::complex<float> const*>(hidden), size);
}

extern "C" void tcm_cRBM_Plus(
    tcm_cRBM const* const x, tcm_cRBM const* const y, tcm_cRBM* const out)
{
    _should_not_throw(
        [](auto const& a, auto const& b, auto& c) { c = a + b; },
        *to_cxx_type(x), *to_cxx_type(y), *to_cxx_type(out));
}

extern "C" void tcm_cRBM_Minus(
    tcm_cRBM const* const x, tcm_cRBM const* const y, tcm_cRBM* const out)
{
    _should_not_throw(
        [](auto const& a, auto const& b, auto& c) { c = a - b; },
        *to_cxx_type(x), *to_cxx_type(y), *to_cxx_type(out));
}

extern "C" void tcm_cRBM_Multiply(
    tcm_cRBM const* const x, tcm_cRBM const* const y, tcm_cRBM* const out)
{
    _should_not_throw(
        [](auto const& a, auto const& b, auto& c) { c = a * b; },
        *to_cxx_type(x), *to_cxx_type(y), *to_cxx_type(out));
}

extern "C" void tcm_cRBM_Divide(
    tcm_cRBM const* const x, tcm_cRBM const* const y, tcm_cRBM* const out)
{
    _should_not_throw(
        [](auto const& a, auto const& b, auto& c) { c = a / b; },
        *to_cxx_type(x), *to_cxx_type(y), *to_cxx_type(out));
}

extern "C" void tcm_cRBM_Negate(
    tcm_cRBM const* const x, tcm_cRBM* const out)
{
    _should_not_throw([](auto const& a, auto& c) { c = -a; },
        *to_cxx_type(x), *to_cxx_type(out));
}
*/

/*
extern "C" void tcm_cMCMC_Destroy(tcm_cMCMC* p)
{
    TRACE();
    if (p != nullptr) { delete p; }
}

extern "C" void tcm_zMCMC_Destroy(tcm_zMCMC* p)
{
    TRACE();
    if (p != nullptr) { delete p; }
}

extern "C" void tcm_cMCMC_Log_wf(
    tcm_cMCMC const* mcmc, tcm_Complex8* out)
{
    TRACE();
    using T = tcm::McmcState<std::complex<float>>;
    *static_cast<std::complex<float>*>(out) =
        static_cast<T const*>(mcmc)->log_wf();
}

extern "C" void tcm_zMCMC_Log_wf(
    tcm_zMCMC const* mcmc, tcm_Complex16* out)
{
    TRACE();
    using T = tcm::McmcState<std::complex<double>>;
    *static_cast<std::complex<double>*>(out) =
        static_cast<T const*>(mcmc)->log_wf();
}

extern "C" void tcm_cMCMC_Log_quotient_wf1(tcm_cMCMC const* mcmc,
    tcm_size_type const flip1, tcm_Complex8* out)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<float>>;
    *static_cast<std::complex<float>*>(out) =
        static_cast<Mcmc const*>(mcmc)->log_quotient_wf(flip1);
}

extern "C" void tcm_zMCMC_Log_quotient_wf1(tcm_zMCMC const* mcmc,
    tcm_size_type const flip1, tcm_Complex16* out)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<double>>;
    *static_cast<std::complex<double>*>(out) =
        static_cast<Mcmc const*>(mcmc)->log_quotient_wf(flip1);
}

extern "C" void tcm_cMCMC_Log_quotient_wf2(tcm_cMCMC const* mcmc,
    tcm_size_type const flip1, tcm_size_type const flip2,
    tcm_Complex8* out)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<float>>;
    *static_cast<std::complex<float>*>(out) =
        static_cast<Mcmc const*>(mcmc)->log_quotient_wf(flip1, flip2);
}

extern "C" void tcm_zMCMC_Log_quotient_wf2(tcm_zMCMC const* mcmc,
    tcm_size_type const flip1, tcm_size_type const flip2,
    tcm_Complex16* out)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<double>>;
    *static_cast<std::complex<double>*>(out) =
        static_cast<Mcmc const*>(mcmc)->log_quotient_wf(flip1, flip2);
}

extern "C" float tcm_cMCMC_Propose1(
    tcm_cMCMC const* mcmc, tcm_size_type const flip1)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<float>>;
    return static_cast<Mcmc const*>(mcmc)->propose(flip1);
}

extern "C" double tcm_zMCMC_Propose1(
    tcm_zMCMC const* mcmc, tcm_size_type const flip1)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<double>>;
    return static_cast<Mcmc const*>(mcmc)->propose(flip1);
}

extern "C" float tcm_cMCMC_Propose2(tcm_cMCMC const* mcmc,
    tcm_size_type const flip1, tcm_size_type const flip2)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<float>>;
    return static_cast<Mcmc const*>(mcmc)->propose(flip1, flip2);
}

extern "C" double tcm_zMCMC_Propose2(tcm_zMCMC const* mcmc,
    tcm_size_type const flip1, tcm_size_type const flip2)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<double>>;
    return static_cast<Mcmc const*>(mcmc)->propose(flip1, flip2);
}

extern "C" void tcm_cMCMC_Accept1(
    tcm_cMCMC* mcmc, tcm_size_type const flip1)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<float>>;
    return static_cast<Mcmc*>(mcmc)->accept(flip1);
}

extern "C" void tcm_zMCMC_Accept1(
    tcm_zMCMC* mcmc, tcm_size_type const flip1)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<double>>;
    return static_cast<Mcmc*>(mcmc)->accept(flip1);
}

extern "C" void tcm_cMCMC_Accept2(tcm_cMCMC* mcmc,
    tcm_size_type const flip1, tcm_size_type const flip2)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<float>>;
    return static_cast<Mcmc*>(mcmc)->accept(flip1, flip2);
}

extern "C" void tcm_zMCMC_Accept2(tcm_zMCMC* mcmc,
    tcm_size_type const flip1, tcm_size_type const flip2)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<double>>;
    return static_cast<Mcmc*>(mcmc)->accept(flip1, flip2);
}

extern "C" void tcm_cMCMC_Print(tcm_cMCMC const* const mcmc)
{
    using Mcmc = tcm::McmcState<std::complex<float>>;
    pretty_print(stdout, static_cast<Mcmc const&>(*mcmc));
    std::fprintf(stdout, "\n");
}

extern "C" void tcm_zMCMC_Print(tcm_zMCMC const* const mcmc)
{
    using Mcmc = tcm::McmcState<std::complex<double>>;
    pretty_print(stdout, static_cast<Mcmc const&>(*mcmc));
    std::fprintf(stdout, "\n");
}

extern "C"
void tcm_cHH1DOpen_Local_energy(tcm_cMCMC const* mcmc, tcm_Complex8* out)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<float>>;
    *static_cast<std::complex<float>*>(out) =
        tcm::heisenberg_1d(static_cast<Mcmc const&>(*mcmc));
}

extern "C"
void tcm_zHH1DOpen_Local_energy(tcm_zMCMC const* mcmc, tcm_Complex16* out)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<double>>;
    *static_cast<std::complex<double>*>(out) =
        tcm::heisenberg_1d(static_cast<Mcmc const&>(*mcmc));
}
*/

/*
extern "C" void tcm_cRBM_heisenberg_1d(tcm_cRBM const* rbm,
    tcm_size_type const offset, tcm_size_type const steps,
    tcm_sMeasurement* out)
{
    TRACE();
    _should_not_throw(
        [](auto const& x, auto const skip, auto const count,
            auto& result) {
            constexpr auto                                  N = 4u;
            tcm::momenta_accumulator<N, float, long double> acc;
            tcm::mcmc_block(x,
                [&acc](auto const& state) {
                    return acc(tcm::heisenberg_1d(state).real());
                },
                skip, count, 0);
            result.mean = acc.get<1>();
            result.var  = acc.get<4>() / std::pow(acc.get<2>(), 2.0f);
        },
        *to_cxx_type(rbm), offset, steps, *out);
}
*/
