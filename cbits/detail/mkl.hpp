#ifndef TCM_SWARM_DETAIL_MKL_HPP
#define TCM_SWARM_DETAIL_MKL_HPP

#include <complex>
#include <type_traits>

// Make sure Intel MKL uses "normal" C++ complex types
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>

#include <mkl.h>

#include "../detail/config.hpp"

TCM_SWARM_BEGIN_NAMESPACE
namespace mkl {

    using size_type       = MKL_UINT;
    using difference_type = MKL_INT;

    static_assert(std::is_same<size_type,
                      std::make_unsigned_t<difference_type>>::value,
        "Expected MKL_UINT to be the \"make_unsigned_t\" version of "
        "MKL_INT.");

    enum class Layout : std::underlying_type_t<CBLAS_LAYOUT> {
        RowMajor = CblasRowMajor,
        ColMajor = CblasColMajor
    };

    enum class Transpose : std::underlying_type_t<CBLAS_TRANSPOSE> {
        None = CblasNoTrans,
        Trans = CblasTrans,
        ConjTrans = CblasConjTrans
    };

    TCM_SWARM_FORCEINLINE
    constexpr auto to_raw_enum(Layout const x) noexcept
        -> CBLAS_LAYOUT
    {
        return static_cast<CBLAS_LAYOUT>(x);
    }

    TCM_SWARM_FORCEINLINE
    constexpr auto to_raw_enum(Transpose const x) noexcept
        -> CBLAS_TRANSPOSE
    {
        return static_cast<CBLAS_TRANSPOSE>(x);
    }

    template <class T, class = void>
    struct real_of;

    template <class T>
    struct real_of<T,
        std::enable_if_t<std::is_floating_point<T>::value>> {
        using type = T;
    };

    template <class T>
    struct real_of<std::complex<T>> {
        using type = T;
    };

    template <class T>
    using real_of_t = typename real_of<T>::type;

} // namespace mkl
TCM_SWARM_END_NAMESPACE

#endif // TCM_SWARM_DETAIL_MKL_HPP
