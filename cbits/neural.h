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

#ifndef TCM_SWARM_NEURAL_H
#define TCM_SWARM_NEURAL_H

#if defined(__cplusplus)
extern "C" {
#endif

#include <stddef.h>

#include <mkl_types.h>

typedef MKL_UINT tcm_size_type;
typedef MKL_INT  tcm_difference_type;

typedef struct tcm_Complex8  tcm_Complex8;
typedef struct tcm_Complex16 tcm_Complex16;

typedef struct tcm_sRBM tcm_sRBM;
typedef struct tcm_dRBM tcm_dRBM;
typedef struct tcm_cRBM tcm_cRBM;
typedef struct tcm_zRBM tcm_zRBM;

typedef struct tcm_sMCMC tcm_sMCMC;
typedef struct tcm_dMCMC tcm_dMCMC;
typedef struct tcm_cMCMC tcm_cMCMC;
typedef struct tcm_zMCMC tcm_zMCMC;

typedef struct tcm_sMeasurement {
    float mean;
    float var;
} tcm_sMeasurement;

// RBM
// ===================================================================

// RbmState<T>::RbmState(size_type, size_type)
// -------------------------------------------------------------------

tcm_sRBM* tcm_sRBM_Create(tcm_size_type const size_visible,
    tcm_size_type const                       size_hidden);
tcm_dRBM* tcm_dRBM_Create(tcm_size_type const size_visible,
    tcm_size_type const                       size_hidden);
tcm_cRBM* tcm_cRBM_Create(tcm_size_type const size_visible,
    tcm_size_type const                       size_hidden);
tcm_zRBM* tcm_zRBM_Create(tcm_size_type const size_visible,
    tcm_size_type const                       size_hidden);

// RbmState<T>::RbmState(RbmState const&)
// -------------------------------------------------------------------

tcm_sRBM* tcm_sRBM_Clone(tcm_sRBM const* rbm);
tcm_dRBM* tcm_dRBM_Clone(tcm_dRBM const* rbm);
tcm_cRBM* tcm_cRBM_Clone(tcm_cRBM const* rbm);
tcm_zRBM* tcm_zRBM_Clone(tcm_zRBM const* rbm);

// RbmState<T>::~RbmState()
// -------------------------------------------------------------------

void tcm_sRBM_Destroy(tcm_sRBM* p);
void tcm_dRBM_Destroy(tcm_dRBM* p);
void tcm_cRBM_Destroy(tcm_cRBM* p);
void tcm_zRBM_Destroy(tcm_zRBM* p);

// RbmState<T>::unsafe_update_weights(T const*)
// -------------------------------------------------------------------

void tcm_sRBM_Set_weights(
    tcm_sRBM* const rbm, float const* const weights);
void tcm_dRBM_Set_weights(
    tcm_dRBM* const rbm, double const* const weights);
void tcm_cRBM_Set_weights(
    tcm_cRBM* const rbm, tcm_Complex8 const* const weights);
void tcm_zRBM_Set_weights(
    tcm_zRBM* const rbm, tcm_Complex16 const* const weights);

// RbmState<T>::unsafe_update_visible(T const*)
// -------------------------------------------------------------------

void tcm_sRBM_Set_visible(
    tcm_sRBM* const rbm, float const* const visible);
void tcm_dRBM_Set_visible(
    tcm_dRBM* const rbm, double const* const visible);
void tcm_cRBM_Set_visible(
    tcm_cRBM* const rbm, tcm_Complex8 const* const visible);
void tcm_zRBM_Set_visible(
    tcm_zRBM* const rbm, tcm_Complex16 const* const visible);

// RbmState<T>::unsafe_update_hidden(T const*)
// -------------------------------------------------------------------

void tcm_sRBM_Set_hidden(
    tcm_sRBM* const rbm, float const* const hidden);
void tcm_dRBM_Set_hidden(
    tcm_dRBM* const rbm, double const* const hidden);
void tcm_cRBM_Set_hidden(
    tcm_cRBM* const rbm, tcm_Complex8 const* const hidden);
void tcm_zRBM_Set_hidden(
    tcm_zRBM* const rbm, tcm_Complex16 const* const hidden);

// RbmState<T>::size_visible()
// -------------------------------------------------------------------

tcm_size_type tcm_sRBM_Size_visible(tcm_sRBM const* const rbm);
tcm_size_type tcm_dRBM_Size_visible(tcm_dRBM const* const rbm);
tcm_size_type tcm_cRBM_Size_visible(tcm_cRBM const* const rbm);
tcm_size_type tcm_zRBM_Size_visible(tcm_zRBM const* const rbm);

// RbmState<T>::size_hidden()
// -------------------------------------------------------------------

tcm_size_type tcm_sRBM_Size_hidden(tcm_sRBM const* const rbm);
tcm_size_type tcm_dRBM_Size_hidden(tcm_dRBM const* const rbm);
tcm_size_type tcm_cRBM_Size_hidden(tcm_cRBM const* const rbm);
tcm_size_type tcm_zRBM_Size_hidden(tcm_zRBM const* const rbm);

// RbmState<T>::pretty_print(FILE*)
// -------------------------------------------------------------------

void tcm_sRBM_Print(tcm_sRBM const* rbm);
void tcm_dRBM_Print(tcm_dRBM const* rbm);
void tcm_cRBM_Print(tcm_cRBM const* rbm);
void tcm_zRBM_Print(tcm_zRBM const* rbm);

// axpby(C, RbmState<T> const&, C, RbmState<T>&)
// -------------------------------------------------------------------

void tcm_sRBM_saxpby(float const alpha, tcm_sRBM const* const x,
    float const beta, tcm_sRBM* const y);
void tcm_dRBM_daxpby(double const alpha, tcm_sRBM const* const x,
    double const beta, tcm_sRBM* const y);
void tcm_cRBM_caxpby(float const alpha_r, float const alpha_i,
    tcm_cRBM const* const x, float const beta_r, float const beta_i,
    tcm_cRBM* const y);
void tcm_zRBM_zaxpby(double const alpha_r, double const alpha_i,
    tcm_zRBM const* const x, double const beta_r, double const beta_i,
    tcm_zRBM* const y);

// scale(C, RbmState<T>&)
// -------------------------------------------------------------------

void tcm_sRBM_sscale(float const alpha, tcm_sRBM* const x);
void tcm_dRBM_dscale(double const alpha, tcm_dRBM* const x);
void tcm_cRBM_cscale(
    float const alpha_r, float const alpha_i, tcm_cRBM* const x);
void tcm_zRBM_zscale(
    double const alpha_r, double const alpha_i, tcm_zRBM* const x);

// delta_well_update(C, RbmState<T>&, RbmState<T> const&, C const*)
// -------------------------------------------------------------------

void tcm_cDelta_well_update(
    float const k, tcm_cRBM* p, tcm_cRBM const* x, float const* u);
void tcm_zDelta_well_update(
    double const k, tcm_zRBM* p, tcm_zRBM const* x, double const* u);

// RBM
// ===================================================================

// McmcState<T>::McmcState(RbmState<T>&, T const*)
// -------------------------------------------------------------------

tcm_sMCMC* tcm_sMCMC_Create(tcm_sRBM* rbm, float const* spin);
tcm_dMCMC* tcm_dMCMC_Create(tcm_dRBM* rbm, double const* spin);
tcm_cMCMC* tcm_cMCMC_Create(tcm_cRBM* rbm, tcm_Complex8 const* spin);
tcm_zMCMC* tcm_zMCMC_Create(tcm_zRBM* rbm, tcm_Complex16 const* spin);

// McmcState<T>::~McmcState()
// -------------------------------------------------------------------

void tcm_sMCMC_Destroy(tcm_sMCMC* p);
void tcm_dMCMC_Destroy(tcm_dMCMC* p);
void tcm_cMCMC_Destroy(tcm_cMCMC* p);
void tcm_zMCMC_Destroy(tcm_zMCMC* p);

// McmcState<T>::log_wf()
// -------------------------------------------------------------------

void tcm_cMCMC_Log_wf(tcm_cMCMC const* mcmc, tcm_Complex8* out);
void tcm_zMCMC_Log_wf(tcm_zMCMC const* mcmc, tcm_Complex16* out);

// McmcState<T>::log_quotient_wf(tcm_size_type...)
// -------------------------------------------------------------------

void tcm_cMCMC_Log_quotient_wf1(tcm_cMCMC const* mcmc,
    tcm_size_type const flip1, tcm_Complex8* out);
void tcm_zMCMC_Log_quotient_wf1(tcm_zMCMC const* mcmc,
    tcm_size_type const flip1, tcm_Complex16* out);

void tcm_cMCMC_Log_quotient_wf2(tcm_cMCMC const* mcmc,
    tcm_size_type const flip1, tcm_size_type const flip2,
    tcm_Complex8* out);
void tcm_zMCMC_Log_quotient_wf2(tcm_zMCMC const* mcmc,
    tcm_size_type const flip1, tcm_size_type const flip2,
    tcm_Complex16* out);

// McmcState<T>::propose(tcm_size_type...)
// -------------------------------------------------------------------

float tcm_cMCMC_Propose1(
    tcm_cMCMC const* mcmc, tcm_size_type const flip1);
double tcm_zMCMC_Propose1(
    tcm_zMCMC const* mcmc, tcm_size_type const flip1);

float  tcm_cMCMC_Propose2(tcm_cMCMC const* mcmc,
     tcm_size_type const flip1, tcm_size_type const flip2);
double tcm_zMCMC_Propose2(tcm_zMCMC const* mcmc,
    tcm_size_type const flip1, tcm_size_type const flip2);

// McmcState<T>::accept(tcm_size_type...)
// -------------------------------------------------------------------

void tcm_cMCMC_Accept1(tcm_cMCMC* mcmc, tcm_size_type const flip1);
void tcm_zMCMC_Accept1(tcm_zMCMC* mcmc, tcm_size_type const flip1);

void tcm_cMCMC_Accept2(tcm_cMCMC* mcmc, tcm_size_type const flip1,
    tcm_size_type const flip2);
void tcm_zMCMC_Accept2(tcm_zMCMC* mcmc, tcm_size_type const flip1,
    tcm_size_type const flip2);

// McmcState<T>::pretty_print(FILE*)
// -------------------------------------------------------------------

void tcm_sMCMC_Print(tcm_sMCMC const* mcmc);
void tcm_dMCMC_Print(tcm_dMCMC const* mcmc);
void tcm_cMCMC_Print(tcm_cMCMC const* mcmc);
void tcm_zMCMC_Print(tcm_zMCMC const* mcmc);

// heisenberg_1d(McmcState<T> const&)
// -------------------------------------------------------------------

void tcm_cHH1DOpen_Local_energy(tcm_cMCMC const* mcmc, tcm_Complex8* out);
void tcm_zHH1DOpen_Local_energy(tcm_zMCMC const* mcmc, tcm_Complex16* out);

void tcm_cRBM_mcmc_block(tcm_cRBM* rbm, tcm_size_type const offset,
    tcm_size_type const steps, tcm_sMeasurement* out);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // TCM_SWARM_NEURAL_H
