
#include <cstdio>
#include <memory>

#define DO_TRACE

#include "neural.h"

#include "RBM.hpp"
#include "momenta.hpp"

// clang format off
struct tcm_sRBM : public tcm::RbmState<float> {};
struct tcm_dRBM : public tcm::RbmState<double> {};
struct tcm_cRBM : public tcm::RbmState<std::complex<float>> {};
struct tcm_zRBM : public tcm::RbmState<std::complex<double>> {};

struct tcm_sMCMC : public tcm::McmcState<float> {};
struct tcm_dMCMC : public tcm::McmcState<double> {};
struct tcm_cMCMC : public tcm::McmcState<std::complex<float>> {};
struct tcm_zMCMC : public tcm::McmcState<std::complex<double>> {};

struct tcm_Complex8  : public std::complex<float> {};
struct tcm_Complex16 : public std::complex<double> {};
// clang-format on

extern "C" tcm_cRBM* tcm_cRBM_Create(
    tcm_size_type const size_visible, tcm_size_type const size_hidden)
{
    TRACE();
    using T = tcm::RbmState<std::complex<float>>;
    try {
        return static_cast<tcm_cRBM*>(
            std::make_unique<T>(size_visible, size_hidden).release());
    }
    catch (std::exception const& e) {
        fprintf(stderr, "[-] Error: tcm_cRBM_Create: %s\n", e.what());
        return nullptr;
    }
}

extern "C" tcm_zRBM* tcm_zRBM_Create(
    tcm_size_type const size_visible, tcm_size_type const size_hidden)
{
    TRACE();
    using T = tcm::RbmState<std::complex<double>>;
    try {
        return static_cast<tcm_zRBM*>(
            std::make_unique<T>(size_visible, size_hidden).release());
    }
    catch (std::exception const& e) {
        fprintf(stderr, "[-] Error: tcm_cRBM_Create: %s\n", e.what());
        return nullptr;
    }
}

extern "C" tcm_cRBM* tcm_cRBM_Clone(tcm_cRBM const* rbm)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
    return static_cast<tcm_cRBM*>(
        std::make_unique<Rbm>(static_cast<Rbm const&>(*rbm))
            .release());
}

extern "C" tcm_zRBM* tcm_zRBM_Clone(tcm_zRBM const* rbm)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<double>>;
    return static_cast<tcm_zRBM*>(
        std::make_unique<Rbm>(static_cast<Rbm const&>(*rbm))
            .release());
}

extern "C" void tcm_cRBM_Destroy(tcm_cRBM* const p)
{
    TRACE();
    if (p != nullptr) { delete p; }
}

extern "C" void tcm_zRBM_Destroy(tcm_zRBM* const p)
{
    TRACE();
    if (p != nullptr) { delete p; }
}

extern "C" void tcm_cRBM_Set_weights(
    tcm_cRBM* const rbm, tcm_Complex8 const* const weights)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
    static_cast<Rbm*>(rbm)->unsafe_update_weights(
        static_cast<Rbm::const_pointer>(weights));
}

extern "C" void tcm_zRBM_Set_weights(
    tcm_zRBM* const rbm, tcm_Complex16 const* const weights)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<double>>;
    static_cast<Rbm*>(rbm)->unsafe_update_weights(
        static_cast<Rbm::const_pointer>(weights));
}

extern "C" void tcm_cRBM_Set_visible(
    tcm_cRBM* const rbm, tcm_Complex8 const* const visible)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
    static_cast<Rbm*>(rbm)->unsafe_update_visible(
        static_cast<Rbm::const_pointer>(visible));
}

extern "C" void tcm_zRBM_Set_visible(
    tcm_zRBM* const rbm, tcm_Complex16 const* const visible)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<double>>;
    static_cast<Rbm*>(rbm)->unsafe_update_visible(
        static_cast<Rbm::const_pointer>(visible));
}

extern "C" void tcm_cRBM_Set_hidden(
    tcm_cRBM* const rbm, tcm_Complex8 const* const hidden)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
    static_cast<Rbm*>(rbm)->unsafe_update_hidden(
        static_cast<Rbm::const_pointer>(hidden));
}

extern "C" void tcm_zRBM_Set_hidden(
    tcm_zRBM* const rbm, tcm_Complex16 const* const hidden)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<double>>;
    static_cast<Rbm*>(rbm)->unsafe_update_hidden(
        static_cast<Rbm::const_pointer>(hidden));
}

extern "C"
tcm_size_type tcm_cRBM_Size_visible(tcm_cRBM const* const rbm)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
    return static_cast<Rbm const*>(rbm)->size_visible();
}

extern "C"
tcm_size_type tcm_zRBM_Size_visible(tcm_zRBM const* const rbm)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<double>>;
    return static_cast<Rbm const*>(rbm)->size_visible();
}

extern "C"
tcm_size_type tcm_cRBM_Size_hidden(tcm_cRBM const* const rbm)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
    return static_cast<Rbm const*>(rbm)->size_hidden();
}

extern "C"
tcm_size_type tcm_zRBM_Size_hidden(tcm_zRBM const* const rbm)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<double>>;
    return static_cast<Rbm const*>(rbm)->size_hidden();
}

extern "C" void tcm_cRBM_Print(tcm_cRBM const* const rbm)
{
    using Rbm = tcm::RbmState<std::complex<float>>;
    pretty_print(stdout, static_cast<Rbm const&>(*rbm));
    std::fprintf(stdout, "\n");
}

extern "C" void tcm_zRBM_Print(tcm_zRBM const* const rbm)
{
    using Rbm = tcm::RbmState<std::complex<double>>;
    pretty_print(stdout, static_cast<Rbm const&>(*rbm));
    std::fprintf(stdout, "\n");
}

void tcm_cRBM_caxpby(float const alpha_r, float const alpha_i,
    tcm_cRBM const* const x, float const beta_r, float const beta_i,
    tcm_cRBM* const y)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
    tcm::axpby({alpha_r, alpha_i}, static_cast<Rbm const&>(*x),
        {beta_r, beta_i}, static_cast<Rbm&>(*y));
}

void tcm_zRBM_zaxpby(double const alpha_r, double const alpha_i,
    tcm_zRBM const* const x, double const beta_r, double const beta_i,
    tcm_zRBM* const y)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<double>>;
    tcm::axpby({alpha_r, alpha_i}, static_cast<Rbm const&>(*x),
        {beta_r, beta_i}, static_cast<Rbm&>(*y));
}

void tcm_cRBM_cscale(
    float const alpha_r, float const alpha_i, tcm_cRBM* const x)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
    tcm::scale({alpha_r, alpha_i}, static_cast<Rbm&>(*x));
}

void tcm_zRBM_zscale(
    double const alpha_r, double const alpha_i, tcm_zRBM* const x)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<double>>;
    tcm::scale({alpha_r, alpha_i}, static_cast<Rbm&>(*x));
}

extern "C" void tcm_cDelta_well_update(
    float const k, tcm_cRBM* p, tcm_cRBM const* x, float const* u)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
    return tcm::delta_well_update(
        k, static_cast<Rbm&>(*p), static_cast<Rbm const&>(*x), u);
}

extern "C" void tcm_zDelta_well_update(
    double const k, tcm_zRBM* p, tcm_zRBM const* x, double const* u)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<double>>;
    return tcm::delta_well_update(
        k, static_cast<Rbm&>(*p), static_cast<Rbm const&>(*x), u);
}

extern "C" tcm_cMCMC* tcm_cMCMC_Create(
    tcm_cRBM* rbm, tcm_Complex8 const* spin)
{
    TRACE();
    using T = tcm::McmcState<std::complex<float>>;
    try {
        return static_cast<tcm_cMCMC*>(std::make_unique<T>(
            *rbm, static_cast<std::complex<float> const*>(spin))
                                           .release());
    }
    catch (std::exception const& e) {
        fprintf(
            stderr, "[-] Error: tcm_cMCMC_Create: %s\n", e.what());
        return nullptr;
    }
}

extern "C" tcm_zMCMC* tcm_zMCMC_Create(
    tcm_zRBM* rbm, tcm_Complex16 const* spin)
{
    TRACE();
    using T = tcm::McmcState<std::complex<double>>;
    try {
        return static_cast<tcm_zMCMC*>(std::make_unique<T>(
            *rbm, static_cast<std::complex<double> const*>(spin))
                                           .release());
    }
    catch (std::exception const& e) {
        fprintf(
            stderr, "[-] Error: tcm_cMCMC_Create: %s\n", e.what());
        return nullptr;
    }
}

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

// extern "C" void tcm_cMCMC_run_mcmc(tcm_cMCMC* mcmc,
//     tcm_size_type const offset, tcm_size_type const steps,
//     int const* const random_ints, float const* const random_floats,
//     tcm_Complex8* out)
// {
//     TRACE();
//     using Mcmc = tcm::McmcState<std::complex<float>>;
//     *static_cast<std::complex<float>*>(out) =
//         tcm::mcmc(static_cast<Mcmc&>(*mcmc), offset, steps,
//             random_ints, random_floats,
//             [](auto const& m){ return tcm::heisenberg_1d(m); },
//             [](auto const& x, auto const& y){ return x + y; });
// }
//
// extern "C" void tcm_zMCMC_run_mcmc(tcm_zMCMC* mcmc,
//     tcm_size_type const offset, tcm_size_type const steps,
//     int const* const random_ints, double const* const random_floats,
//     tcm_Complex16* out)
// {
//     TRACE();
//     using Mcmc = tcm::McmcState<std::complex<double>>;
//     *static_cast<std::complex<double>*>(out) =
//         tcm::mcmc(static_cast<Mcmc&>(*mcmc), offset, steps,
//             random_ints, random_floats,
//             [](auto const& m){ return tcm::heisenberg_1d(m); },
//             [](auto const& x, auto const& y){ return x + y; });
// }

extern "C" void tcm_cRBM_mcmc_block(tcm_cRBM* rbm,
    tcm_size_type const offset, tcm_size_type const steps,
    tcm_sMeasurement* out)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;

    constexpr auto N = 2u;
    tcm::momenta_accumulator<N, float, long double> acc;
    try {
    tcm::mcmc_block(static_cast<Rbm&>(*rbm),
        [&acc](auto& m) { return acc(tcm::heisenberg_1d(m).real()); },
        offset, steps, 2);
    }
    catch (std::exception& e) {
        std::fprintf(stderr, "Exception: %s\n", e.what());
        std::terminate();
    }
    out->mean = acc.get<1>();
    out->var  = acc.get<2>();
}

