
#include <cstdio>
#include <memory>

#include "neural.h"

#include "RBM.hpp"

struct tcm_cRBM : public tcm::RbmState<std::complex<float>> {
};

struct tcm_cMCMC : public tcm::McmcState<std::complex<float>> {
};

struct tcm_Complex8 : public std::complex<float> {
};

struct tcm_Complex16 : public std::complex<double> {
};

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

extern "C" void tcm_cRBM_Destroy(tcm_cRBM* const p)
{
    TRACE();
    if (p != nullptr) { delete p; }
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

extern "C" void tcm_cMCMC_Destroy(tcm_cMCMC* p)
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

extern "C" void tcm_cMCMC_Log_quotient_wf1(tcm_cMCMC const* mcmc,
    tcm_size_type const flip1, tcm_Complex8* out)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<float>>;
    *static_cast<std::complex<float>*>(out) =
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

extern "C" float tcm_cMCMC_Propose1(
    tcm_cMCMC const* mcmc, tcm_size_type const flip1)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<float>>;
    return static_cast<Mcmc const*>(mcmc)->propose(flip1);
}

extern "C" void tcm_cMCMC_Accept1(
    tcm_cMCMC* mcmc, tcm_size_type const flip1)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<float>>;
    return static_cast<Mcmc*>(mcmc)->accept(flip1);
}

extern "C" float tcm_cMCMC_Propose2(tcm_cMCMC const* mcmc,
    tcm_size_type const flip1, tcm_size_type const flip2)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<float>>;
    return static_cast<Mcmc const*>(mcmc)->propose(flip1, flip2);
}

extern "C" void tcm_cMCMC_Accept2(tcm_cMCMC* mcmc,
    tcm_size_type const flip1, tcm_size_type const flip2)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<float>>;
    return static_cast<Mcmc*>(mcmc)->accept(flip1, flip2);
}

extern "C" void tcm_cRBM_Print(tcm_cRBM const* const rbm)
{
    using Rbm = tcm::RbmState<std::complex<float>>;
    pretty_print(stdout, static_cast<Rbm const&>(*rbm));
    std::fprintf(stdout, "\n");
}

extern "C" void tcm_cMCMC_Print(tcm_cMCMC const* const mcmc)
{
    using Mcmc = tcm::McmcState<std::complex<float>>;
    pretty_print(stdout, static_cast<Mcmc const&>(*mcmc));
    std::fprintf(stdout, "\n");
}

extern "C" void tcm_cRBM_Set_weights(
    tcm_cRBM* const rbm, tcm_Complex8 const* const weights)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
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

extern "C" void tcm_cRBM_Set_hidden(
    tcm_cRBM* const rbm, tcm_Complex8 const* const hidden)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
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
tcm_size_type tcm_cRBM_Size_hidden(tcm_cRBM const* const rbm)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
    return static_cast<Rbm const*>(rbm)->size_hidden();
}

extern "C" tcm_cRBM* tcm_cRBM_Clone(tcm_cRBM const* rbm)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
    return static_cast<tcm_cRBM*>(
        std::make_unique<Rbm>(static_cast<Rbm const&>(*rbm))
            .release());
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

void tcm_cRBM_cscale(
    float const alpha_r, float const alpha_i, tcm_cRBM* const x)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
    tcm::scale({alpha_r, alpha_i}, static_cast<Rbm&>(*x));
}

extern "C"
void tcm_cHH1DOpen_Local_energy(tcm_cMCMC const* mcmc, tcm_Complex8* out)
{
    TRACE();
    using Mcmc = tcm::McmcState<std::complex<float>>;
    *static_cast<std::complex<float>*>(out) =
        tcm::heisenberg_1d(static_cast<Mcmc const&>(*mcmc));
}

extern "C" void tcm_cDelta_well_update(
    float const k, tcm_cRBM* p, tcm_cRBM const* x, float const* u)
{
    TRACE();
    using Rbm = tcm::RbmState<std::complex<float>>;
    return tcm::delta_well_update(
        k, static_cast<Rbm&>(*p), static_cast<Rbm const&>(*x), u);
}
