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

#ifndef TCM_SWARM_NQS_H
#define TCM_SWARM_NQS_H

#if defined(__cplusplus)
extern "C" {
#endif

#include "nqs_types.h"

/**
 * In-place constructs an RBM with a given number of visible and hidden units.
 *
 * \param self A pointer to at least #TCM_SWARM_SIZEOF_RBM bytes aligned to
 *             #TCM_SWARM_ALIGNOF_RBM.
 * \param size_visible Number of visible units. Must be at least 0.
 * \param size_hidden  Number of hidden units. Must be at least 0.
 */
void tcm_Rbm_create(tcm_Rbm* self, tcm_Index size_visible, tcm_Index size_hidden);

/**
 * Destructs the RBM.
 *
 * \param self A pointer to the RBM. Must not be null.
 */
void tcm_Rbm_destroy(tcm_Rbm* self);

/**
 * Returns the total number of tunable parameters in the given RBM.
 *
 * \param self A pointer to the RBM. Must not be null.
 */
tcm_Index tcm_Rbm_size(tcm_Rbm const* self);

/**
 * Returns the number of visible units in the given RBM.
 *
 * \param self A pointer to the RBM. Must not be null.
 */
tcm_Index tcm_Rbm_size_visible(tcm_Rbm const* self);

/**
 * Returns the number of hidden units in the given RBM.
 *
 * \param self A pointer to the RBM. Must not be null.
 */
tcm_Index tcm_Rbm_size_hidden(tcm_Rbm const* self);

/**
 * Returns the number of weights in the given RBM. This is equivalent to calling
 * `tcm_Rbm_size_visible(self) * tcm_Rbm_size_hidden(self)`.
 *
 * \param self A pointer to the RBM. Must not be null.
 */
tcm_Index tcm_Rbm_size_weights(tcm_Rbm const* self);

void tcm_Rbm_get_visible(tcm_Rbm* self, tcm_Vector* out);

void tcm_Rbm_get_hidden(tcm_Rbm* self, tcm_Vector* out);

void tcm_Rbm_get_weights(tcm_Rbm* self, tcm_Matrix* out);

void tcm_Heisenberg_create(tcm_Hamiltonian* self, tcm_Index (*edges)[2],
    tcm_Index const n, int const t1, int const t2, float const* cutoff);
void tcm_Heisenberg_destroy(tcm_Hamiltonian* self);

void tcm_sample_moments(tcm_Rbm const* self, tcm_Hamiltonian const* hamiltonian,
    tcm_MC_Config const* config, tcm_Vector* moments, tcm_MC_Stats* stats);

void tcm_sample_gradients(tcm_Rbm const* self,
    tcm_Hamiltonian const* hamiltonian, tcm_MC_Config const* config,
    tcm_Vector* moments, tcm_Vector* force, tcm_Matrix* gradients,
    tcm_MC_Stats* stats);

void tcm_set_log_level(tcm_Level level);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // TCM_SWARM_NQS_H
