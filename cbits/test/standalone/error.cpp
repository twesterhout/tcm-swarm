

#include <stdexcept>
#include <iostream>

#include "../../detail/errors.hpp"

extern auto foo();

auto foo()
{
    tcm::throw_with_trace(std::runtime_error{"Haha!"});
}

int main()
{
    try {
        foo();
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << '\n';
        auto const* st = boost::get_error_info<tcm::traced>(e);
        if (st != nullptr) {
            std::cerr << "Backtrace:\n" << *st << '\n';
        }
    }
}
