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

#ifndef TCM_SWARM_DETAIL_RANDOM_HPP
#define TCM_SWARM_DETAIL_RANDOM_HPP

#include <complex>
#include <random>

#include <gsl/gsl>

#include "../spin.hpp"

#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wdocumentation"
#elif defined(TCM_SWARM_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wnoexcept"
#endif

#include <range/v3/view/generate.hpp>

#if defined(TCM_SWARM_CLANG)
#pragma clang diagnostic pop
#elif defined(TCM_SWARM_GCC)
#pragma GCC diagnostic pop
#endif

TCM_SWARM_BEGIN_NAMESPACE

auto really_need_that_random_seed_now() -> std::uint64_t;
auto& thread_local_generator();

/// \rst
/// Returns a "truly" random seed (generated using `random_device`_).
///
/// .. note::
///
///    Please, use this function sparingly as it is probably very slow.
///
/// \endrst
TCM_SWARM_NOINLINE
auto really_need_that_random_seed_now() -> std::uint64_t
{
    static thread_local std::random_device random_device;
    std::uniform_int_distribution<std::uint64_t> dist;
    return dist(random_device);
}

/// \rst
/// Returns a reference to current thread's `UniformRandomBitGenerator`_.
/// \endrst
TCM_SWARM_NOINLINE
auto& thread_local_generator()
{
    static thread_local std::mt19937 generator{really_need_that_random_seed_now()};
    return generator;
}

/// \rst
/// Creates a random spin configuration of given size, where spins are
/// repsesented by complex numbers: :math:`\downarrow \equiv -1 + 0i` and
/// :math:`\uparrow \equiv 1 + 0i`.
/// \endrst
template <class VectorType, class IndexType,
    class Generator = std::decay_t<decltype(thread_local_generator())>,
    class           = std::enable_if_t<std::is_integral_v<IndexType>>>
auto make_random_spin(IndexType const n,
    Generator& gen = thread_local_generator()) -> VectorType
{
    using std::begin, std::end;
    using C         = typename VectorType::value_type;
    using R         = typename C::value_type;
    using size_type = typename VectorType::size_type;
    if (n < 0) {
        throw std::domain_error{
            "Can't create a spin configuration with a negative ("
            + std::to_string(n) + ") number of spins."};
    }
    if (n == 0) { return {}; }

    VectorType spin(gsl::narrow_cast<size_type>(n));
    std::uniform_int_distribution<int> dist{0, 1};
    std::generate(begin(spin), end(spin), [&gen, &dist]() -> C {
        return {R{2} * gsl::narrow<R>(dist(gen)) - R{1}, 0};
    });
    Ensures(TCM_SWARM_IS_VALID_SPIN(gsl::span<C const>{spin}));
    return spin;
}

template <class VectorType, class IndexType,
    class Generator = std::decay_t<decltype(thread_local_generator())>,
    class           = std::enable_if_t<std::is_integral_v<IndexType>>>
auto make_random_spin(IndexType const n, IndexType const magnetisation,
    Generator& gen = thread_local_generator()) -> VectorType
{
    using std::begin, std::end;
    using C         = typename VectorType::value_type;
    using R         = typename C::value_type;
    using size_type = typename VectorType::size_type;
    if (n < 0) {
        throw std::domain_error{
            "Can't create a spin configuration with a negative ("
            + std::to_string(n) + ") number of spins."};
    }
    if (std::abs(magnetisation) > n) {
        throw std::domain_error{
            std::to_string(n)
            + " spins can't have a total magnetisation of "
            + std::to_string(magnetisation) + "."};
    }
    if ((n + magnetisation) % 2 != 0) {
        throw std::domain_error{
            std::to_string(n)
            + " spins can't have a total magnetisation of "
            + std::to_string(magnetisation)
            + ". `n + magnetisation` must be even."};
    }
    if (n == 0) { return {}; }
    auto const number_ups   = (n + magnetisation) / 2;
    auto const number_downs = (n - magnetisation) / 2;
    VectorType spin(gsl::narrow_cast<size_type>(n));
    std::fill_n(begin(spin), number_ups, C{1});
    std::fill_n(begin(spin) + number_ups, number_downs, C{-1});
    std::shuffle(begin(spin), end(spin), gen);
    Ensures(TCM_SWARM_CHECK_MAGNETISATION(
        gsl::span<C const>{spin}, magnetisation));
    return spin;
}

#if 0
template <class R, class Generator>
auto uniform_float_stream(Generator& generator = thread_local_generator())
{
    return ranges::view::generate([&generator]() -> R {
        std::uniform_real_distribution<R> dist{0, 1};
        return dist(generator);
    });
}
#endif

#if 0
namespace mkl {

enum class GenType : MKL_INT {
    mcg31         = VSL_BRNG_MCG31,
    r250          = VSL_BRNG_R250,
    mrg32k3a      = VSL_BRNG_MRG32K3A,
    mcg59         = VSL_BRNG_MCG59,
    wh            = VSL_BRNG_WH,
    sobol         = VSL_BRNG_SOBOL,
    niederr       = VSL_BRNG_NIEDERR,
    mt19937       = VSL_BRNG_MT19937,
    mt2203        = VSL_BRNG_MT2203,
    // iabstract     = VSL_BRNG_IABSTRACT,
    // dabstract     = VSL_BRNG_DABSTRACT,
    // sabstract     = VSL_BRNG_SABSTRACT,
    sfmt19937     = VSL_BRNG_SFMT19937,
    // nondeterm     = VSL_BRNG_NONDETERM,
    // ars5          = VSL_BRNG_ARS5,
    philox4x32x10 = VSL_BRNG_PHILOX4X32X10
};

template <class T>
constexpr auto swap(T*& x, T*& y) noexcept -> void
{
    auto const t = x;
    x            = y;
    y            = t;
}

struct random_generator {
    using state_ptr_t = VSLStreamStatePtr;
    static_assert(std::is_pointer_v<state_ptr_t>,
        "VSLStreamStatePtr is not a pointer.");

  private:
    [[nodiscard]] static auto _make_generator(GenType const generator,
        size_type const seed) -> gsl::owner<state_ptr_t>
    {
        state_ptr_t p;
        auto const  err = vslNewStream(&p,
            static_cast<std::underlying_type_t<GenType>>(generator), seed);
        if (TCM_SWARM_UNLIKELY(err != VSL_ERROR_OK)) {
            switch (err) {
            case VSL_ERROR_MEM_FAILURE: throw std::bad_alloc{};
            case VSL_RNG_ERROR_NONDETERM_NOT_SUPPORTED:
                throw std::runtime_error{
                    "tcm::mkl::random_generator::_make_generator(): "
                    "Non-deterministic generator is not supported."};
            case VSL_RNG_ERROR_ARS5_NOT_SUPPORTED:
                throw std::runtime_error{
                    "tcm::mkl::random_generator::_make_generator(): "
                    "ARS5 generator is not supported."};
            }
        }
        Ensures(p != nullptr);
        return p;
    }

  public:
    explicit random_generator(GenType const generator, size_type const seed)
        : _handle{_make_generator(generator, seed)}
    {
        Ensures(_handle != nullptr);
    }

    random_generator(random_generator const&) = delete;

    constexpr random_generator(random_generator&& other) noexcept(
        !tcm::detail::gsl_can_throw())
        : _handle{other._handle}
    {
        Expects(other._handle != nullptr);
        other._handle = nullptr;
        Ensures(_handle != nullptr);
    }

    random_generator& operator=(random_generator const&) = delete;
    random_generator& operator=(random_generator&&) noexcept = delete;

    constexpr auto get() -> gsl::not_null<state_ptr_t>
    {
        return _handle;
    }

    ~random_generator()
    {
        if (_handle != nullptr) {
            vslDeleteStream(&_handle);
        }
    }

  private:
    state_ptr_t _handle;
};

enum class Method : MKL_INT {
    uniform_standard = VSL_RNG_METHOD_UNIFORM_STD,
    uniform_accurate = VSL_RNG_METHOD_UNIFORM_STD_ACCURATE,
};

namespace detail {
    // clang-format off
    [[nodiscard]] TCM_SWARM_FORCEINLINE
    auto uniform(std::underlying_type_t<Method> const method,
        random_generator::state_ptr_t const stream,
        difference_type const n, float* const r, float const a,
        float const b) noexcept
    // clang-format on
    {
        return vsRngUniform(method, stream, n, r, a, b);
    }

    // clang-format off
    template <GenType generator>
    [[nodiscard]] TCM_SWARM_FORCEINLINE
    auto uniform(std::underlying_type_t<Method> const method,
        random_generator::state_ptr_t const stream,
        difference_type const n, double* const r, double const a,
        double const b) noexcept
    // clang-format on
    {
        return vdRngUniform(method, stream, n, r, a, b);
    }

    // clang-format off
    [[nodiscard]] TCM_SWARM_FORCEINLINE
    auto uniform(std::underlying_type_t<Method> const method,
        random_generator::state_ptr_t const stream,
        difference_type const n, int* const r, int const a,
        int const b) noexcept
    // clang-format on
    {
        return viRngUniform(method, stream, n, r, a, b);
    }

    // clang-format off
    [[nodiscard]] TCM_SWARM_FORCEINLINE
    auto uniform(std::underlying_type_t<Method> const method,
        random_generator::state_ptr_t const stream,
        difference_type const n, std::complex<float>* const r,
        float const a, float const b) noexcept
    // clang-format on
    {
        static_assert(sizeof(std::complex<float>) == 2 * sizeof(float));
        return vsRngUniform(
            method, stream, 2 * n, reinterpret_cast<float*>(r), a, b);
    }
} // namespace detail

struct uniform_fn {
    template <class T, class U>
    auto operator()(Method const method, random_generator& stream,
        gsl::span<T> const r, U const a, U const b) const -> void
    {
        auto const err = detail::uniform(
            static_cast<std::underlying_type_t<Method>>(method),
            stream.get(), static_cast<difference_type>(r.size()), r.data(),
            a, b);
        if (TCM_SWARM_UNLIKELY(err != VSL_ERROR_OK)) {
            if (err == VSL_RNG_ERROR_QRNG_PERIOD_ELAPSED) {
                throw std::runtime_error{
                    "tcm::mkl::uniform_fn::operator(): "
                    "Period of the generator has been exceeded."};
            }
            else { // This part should in principle never be executed.
                throw std::runtime_error{
                    "tcm::mkl::uniform_fn::operator(): Bug! "
                    "Unreachable code reached!"};
            }
        }
    }
};

TCM_SWARM_INLINE_VARIABLE(uniform_fn, uniform)

} // namespace mkl
#endif

#if 0
/// \brief Infinite stream generated by repreated applicatoin of a
/// function.
///
/// Internally, we keep an array of `T`s of length `BlockSize` aligned to
/// `Alignment`. Then, instead of generating values one by one, they are created
/// in batches of `BlockSize` elements at once. This allows for SIMD-related
/// optimisations.
///
/// \tparam T         type of the elements.
/// \tparam BlockSize length of the internal buffer.
/// \tparam Alignment alignment of the internal buffer (this is useful for
///                   SIMD).
template <class T, std::size_t BlockSize, std::size_t Alignment>
struct buffered_generator
    : public ranges::view_facade<buffered_generator<T, BlockSize, Alignment>> {

    static_assert(std::is_same_v<T, std::decay_t<T>>,
        "Please, use a normal type :)");

    using buffered_generator_type =
        buffered_generator<T, BlockSize, Alignment>;
    using value_type           = T;
    using reference            = T&;
    using const_reference      = T const&;
    using refill_function_type = auto(gsl::span<value_type>) -> void;

    /// \brief Returns the length of internal buffer.
    static constexpr auto block_size() noexcept { return BlockSize; }

    /// \brief Returns the alignment of internal buffer in bytes.
    static constexpr auto alignment() noexcept { return Alignment; }

  private:
    friend ranges::range_access;

    std::function<refill_function_type> _refill;
    alignas(alignment()) std::array<value_type, block_size()> _block;

    auto refill()
    {
        Expects(
            reinterpret_cast<std::uintptr_t>(_block.data()) % alignment()
            == 0u);
        _refill(gsl::make_span(_block));
    }

  public:
    /// \brief Given a "refiller" function constructs a new buffered
    /// generator.
    ///
    /// This constructor is eager in the sense that the internal buffer is
    /// filled upon construction rather than upon the first request.
    ///
    /// \param fn A function of type `auto (gsl::span<value_type>) -> void`
    /// which given a buffer, fills it with some values.
    template <class RefillFunction,
        class = std::enable_if_t<std::is_constructible_v<
            std::function<refill_function_type>, RefillFunction&&>>>
    buffered_generator(RefillFunction&& fn)
        : _refill{std::forward<RefillFunction>(fn)}
    {
        refill();
    }

    buffered_generator(buffered_generator const&) = default;
    buffered_generator(buffered_generator&&) = default;
    buffered_generator& operator=(buffered_generator const&) = default;
    buffered_generator& operator=(buffered_generator&&) = default;

  private:
    /// \brief A kind of stripped down iterator over the random_range.
    ///
    /// This class is quite lightweight -- it only keeps track of the current
    /// position in the buffer. When the end of buffer is reached, it instructs
    /// the range to refill the buffer which provides an illusion of an infinite
    /// stream.
    struct cursor {
        buffered_generator_type* _stream;
        std::size_t              _i;

        decltype(auto) read() const noexcept(!detail::gsl_can_throw())
        {
            Expects(_stream != nullptr);
            Expects(_i < buffered_generator_type::block_size());
            return _stream->_block[_i];
        }

        auto equal(ranges::default_sentinel /*unused*/) const
            noexcept(!detail::gsl_can_throw())
        {
            return false;
        }

        auto next() -> void
        {
            Expects(_stream != nullptr);
            Expects(_i < buffered_generator_type::block_size());
            if (++_i == buffered_generator_type::block_size()) {
                _stream->refill();
                _i = 0;
            }
            Ensures(_i < buffered_generator_type::block_size());
        }
    };

    constexpr auto begin_cursor() noexcept -> cursor
    {
        return {this, 0u};
    }
};

/// \brief Metafunction returning a standard uniform distribution corresponding
/// to `T`.
template <class T, class = void>
struct uniform_distribution;

template <class T>
struct uniform_distribution<T,
    std::enable_if_t<std::is_integral_v<T>>> {
    using type = std::uniform_int_distribution<T>;
};

template <class T>
struct uniform_distribution<T,
    std::enable_if_t<std::is_floating_point_v<T>>> {
    using type = std::uniform_real_distribution<T>;
};

template <class T>
using uniform_distribution_t = typename uniform_distribution<T>::type;

/// \brief Given a UniformRandomBitGenerator and lower and upper bounds, creates
/// an infinite range of random numbers of type `T` distributed uniformly in
/// `[min, max)`.
template <class T, std::size_t BlockSize, std::size_t Alignment, class Generator>
auto make_std_random_stream(Generator& g, T const min, T max)
{
    struct refill_fn {
        Generator&                _gen;
        uniform_distribution_t<T> _dist;

        auto operator()(gsl::span<T> const x) noexcept(
            noexcept(std::declval<uniform_distribution_t<T>&>()(
                std::declval<Generator&>()))) -> void
        {
            std::generate(std::begin(x), std::end(x),
                [this]() { return _dist(_gen); });
        }
    };
    if constexpr (std::is_integral_v<T>) {
        Expects(max > min);
        --max;
    }
    return buffered_generator<T, BlockSize, Alignment>{
        refill_fn{g, uniform_distribution_t<T>{min, max}}};
}
#endif




TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_RANDOM_HPP

