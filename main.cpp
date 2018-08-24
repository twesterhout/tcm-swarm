#include <iostream>
#include <chrono>
#include <thread>
#include <sstream>

#include <memory>

struct deleter {
    template <class T>
    auto operator()(T* p) noexcept
    {
        std::free(p);
    };
};

auto foo(int n)
{
    auto* p = std::aligned_alloc(64, n * sizeof(float));
    return std::unique_ptr<float[], deleter>{static_cast<float*>(p)};
}

auto bar(int n)
{
    auto x = foo(n);
    x[0] = 10.0f;
}

int main()
{
    bar(123);
}
