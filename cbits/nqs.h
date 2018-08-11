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
typedef int      tcm_index_t;

typedef struct _tcm_Complex8 {
    float real;
    float imag;
} tcm_Complex8;

typedef struct _tcm_Complex16 {
    double real;
    double imag;
} tcm_Complex16;

typedef struct _tcm_cRbm tcm_cRbm;
typedef struct _tcm_zRbm tcm_zRbm;

typedef enum _tcm_Hamiltonian {
    HEISENBERG_1D_OPEN,
    HEISENBERG_1D_PERIODIC
} tcm_Hamiltonian;

typedef struct _tcm_Range {
    int start;
    int stop;
    int step;
} tcm_Range;

typedef struct _tcm_cEstimate {
    tcm_Complex8 value;
    tcm_Complex8 error;
} tcm_cEstimate;

typedef struct _tcm_cTensor1 {
    tcm_Complex8* data;
    int           extents[1];
} tcm_cTensor1;

typedef struct _tcm_cTensor2 {
    tcm_Complex8* data;
    int           extents[2];
} tcm_cTensor2;

typedef struct _tcm_cSRMeasurement {
    tcm_Complex8* force;
    tcm_Complex8* derivatives;

};

// RBM
// ===================================================================

// Rbm<T>::Rbm(size_type, size_type)
// -------------------------------------------------------------------

tcm_cRbm* tcm_cRbm_Create(int const size_visible, int const size_hidden);
tcm_zRbm* tcm_zRbm_Create(int const size_visible, int const size_hidden);

tcm_cRbm* tcm_cRbm_Clone(tcm_cRbm const* const rbm);
tcm_zRbm* tcm_zRbm_Clone(tcm_zRbm const* const rbm);

int tcm_cRbm_Size(tcm_cRbm const* const rbm);
int tcm_cRbm_Size_visible(tcm_cRbm const* const rbm);
int tcm_cRbm_Size_hidden(tcm_cRbm const* const rbm);

void tcm_cRbm_Destroy(tcm_cRbm* const p);
void tcm_zRbm_Destroy(tcm_zRbm* const p);

void tcm_cRbm_Get_visible(tcm_cRbm* const rbm, tcm_cTensor1* const visible);
void tcm_zRbm_Get_visible(tcm_zRbm* const rbm, tcm_cTensor1* const visible);

void tcm_cRbm_Get_hidden(tcm_cRbm* const rbm, tcm_cTensor1* const hidden);
void tcm_zRbm_Get_hidden(tcm_zRbm* const rbm, tcm_cTensor1* const hidden);

void tcm_cRbm_Get_weights(tcm_cRbm* const rbm, tcm_cTensor2* const weights);
void tcm_zRbm_Get_weights(tcm_zRbm* const rbm, tcm_cTensor2* const weights);

void tcm_cRbm_Sample(
    tcm_cRbm const* const rbm,
    tcm_Hamiltonian const hamiltonian,
    int const number_runs,
    tcm_Range const* const steps,
    int const* const magnetisation,
    tcm_cEstimate* energy,
    tcm_cTensor1* force,
    tcm_cTensor2* derivatives);

void tcm_cRbm_Sample_Moments(tcm_cRbm const* const rbm,
    tcm_Hamiltonian const hamiltonian, int const number_runs,
    tcm_Range const* const steps, int const* const magnetisation,
    tcm_cTensor1* moments);

void tcm_cCovariance(tcm_cTensor2 const* derivatives, tcm_cTensor2* out);

void tcm_cRbm_Axpy(tcm_Complex8 const* a, tcm_cTensor1 const* x, tcm_cRbm* y);


/*
void tcm_cRBM_Set_weights(tcm_cRBM* const rbm,
    tcm_Complex8 const* const weights, tcm_size_type const size);
void tcm_zRBM_Set_weights(tcm_zRBM* const rbm,
    tcm_Complex16 const* const weights, tcm_size_type const size);

void tcm_cRBM_Set_visible(tcm_cRBM* const rbm,
    tcm_Complex8 const* const visible, tcm_size_type const size);
void tcm_zRBM_Set_visible(tcm_zRBM* const rbm,
    tcm_Complex16 const* const visible, tcm_size_type const size);

void tcm_cRBM_Set_hidden(tcm_cRBM* const rbm,
    tcm_Complex8 const* const hidden, tcm_size_type const size);
void tcm_zRBM_Set_hidden(tcm_zRBM* const rbm,
    tcm_Complex16 const* const hidden, tcm_size_type const size);
*/

/*
void tcm_cRBM_Plus(
    tcm_cRBM const* const x, tcm_cRBM const* const y, tcm_cRBM* const out);
void tcm_cRBM_Minus(
    tcm_cRBM const* const x, tcm_cRBM const* const y, tcm_cRBM* const out);
void tcm_cRBM_Negate(tcm_cRBM const* const x, tcm_cRBM* const out);
void tcm_cRBM_Multiply(
    tcm_cRBM const* const x, tcm_cRBM const* const y, tcm_cRBM* const out);
void tcm_cRBM_Divide(
    tcm_cRBM const* const x, tcm_cRBM const* const y, tcm_cRBM* const out);
*/

/*

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
*/


#if defined(__cplusplus)
} // extern "C"
#endif

#endif // TCM_SWARM_NEURAL_H
