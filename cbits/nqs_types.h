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

#ifndef TCM_SWARM_NQS_TYPES_H
#define TCM_SWARM_NQS_TYPES_H

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(C2HS_IN_HOUSE)
#include <stdbool.h>
#endif

#include <stddef.h>

typedef ptrdiff_t tcm_Index;

#if !defined(C2HS_IN_HOUSE)
typedef _Complex float tcm_Complex;
#else // c2hs doesn't work well with C99 complex numbers.
typedef struct {
    float real;
    float imag;
} tcm_Complex;
#endif

/**
 * A non-owning vector suitable for BLAS operations.
 */
typedef struct tcm_Vector {
    /**
     * A non-owning pointer to the buffer.
     */
    void* data;

    /**
     * Number of elements in the vector. Assumed to be at least zero.
     */
    int size;

    /**
     * Distance between two consecutive elements. Assumed to be at least one.
     */
    int stride;
} tcm_Vector;

/**
 * A non-owning matrix suitable for BLAS operations.
 *
 * Currently padded with 4 bytes. This space could be used to store the layout.
 * This may be done in the future.
 */
typedef struct tcm_Matrix {
    /**
     * A non-owning pointer to the buffer.
     */
    void* data;

    /**
     * Number of rows in the matrix. Assumed to be at least zero.
     */
    int rows;

    /**
     * Number of columns in the matrix. Assumed to be at least zero.
     */
    int cols;

    /**
     * Distance between two consecutive rows of columns depending on the layout.
     */
    int stride;
} tcm_Matrix;

#define TCM_SWARM_SIZEOF_RBM 64
#define TCM_SWARM_ALIGNOF_RBM 8

/**
 * Opaque struct for representing tcm::RbmBase<std::complex<float>>.
 *
 * The only information provided is that a tcm_Rbm fits into TCM_SWARM_SIZEOF_RBM bytes
 * and its alignment it no stricted than TCM_SWARM_ALIGNOF_RBM.
 */
typedef struct _tcm_Rbm tcm_Rbm;

/**
 * Isotropic Heisenberg Hamiltonian.
 */
typedef struct _tcm_Heisenberg tcm_Heisenberg;

/**
 * Types of Hamiltonians supported.
 */
typedef enum _tcm_HamiltonianType {
    TCM_SPIN_HEISENBERG,             ///< Isotropic Heisenberg Hamiltonian.
    TCM_SPIN_ANISOTROPIC_HEISENBERG, ///< Anysotropic Heisenberg Hamiltonian.
    TCM_SPIN_DZYALOSHINSKII_MORIYA,  ///< Dzyaloshingkii-Moriya Hamiltonian.
} tcm_HamiltonianType;

/**
 * Polymorphic representation of a Hamiltonian.
 */
typedef struct _tcm_Hamiltonian {
    void*               payload;
    tcm_HamiltonianType dtype;
} tcm_Hamiltonian;

/* Taken from spdlog:
enum level_enum
{
    trace = 0,
    debug = 1,
    info = 2,
    warn = 3,
    err = 4,
    critical = 5,
    off = 6
};
*/
typedef enum _tcm_Level {
    tcm_Level_trace    = 0,
    tcm_Level_debug    = 1,
    tcm_Level_info     = 2,
    tcm_Level_warn     = 3,
    tcm_Level_err      = 4,
    tcm_Level_critical = 5,
    tcm_Level_off      = 6
} tcm_Level;

/**
 * Options for running Monte-Carlo sampling.
 */
typedef struct tcm_MC_Config {
    /**
     * Specifies the indices of elements in the Markov chain
     * which are used for sampling. `range` has the following form:
     * `{start, stop, step}`.  Elements at indices
     * `start, start + step, start + 2 * step, ...` will be used for sampling.
     * The `stop` index is *not* included. This is very similar to how
     * Python's `range` function works.
     */
    tcm_Index range[3];
    /**
     * Number of threads to use at each level of the computation.
     */
    int threads[3];

    /**
     * Number of independent Monte-Carlo simulations to average over.
     */
    int runs;

    /**
     * Number of spin-flips to do at each step in the Markov chain.
     */
    int flips;

    /**
     * Sometimes, when calculating Hamiltonians, it turns out
     * that the current spin configuration has negligible weight
     * in the wave function. In such cases it makes sense to restart
     * the Monte-Carlo sampling from a different spin configuration.
     * `restarts` specifies the maximal number of restarts permitted per run.
     */
    int restarts;

    /**
     * Specifies whether magnetisation should be conserved.
     */
    // TODO: This is not very pretty...
    bool has_magnetisation;

    /**
     * If `has_magnetisation == true` then specifies the
     * value of the magnetisation. Otherwise, it undefined.
     */
    int magnetisation;
} tcm_MC_Config;

struct _tcm_MC_Stats {
    /**
     * Effective dimension of the sampled Hilbert space, i.e. the number of
     * basis states visited during Monte-Carlo sampling.
     */
    tcm_Index dimension;

    /**
     * Variance in E. If no variance could be computed, a special value (-1 + 0*I) is used instead.
     */
    float variance;
};

typedef struct _tcm_MC_Stats tcm_MC_Stats;

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // TCM_SWARM_NQS_TYPES_H
