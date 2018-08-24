
#include "cbits/logging.hpp"

int main()
{
    auto l = tcm::global_logger();
    l->warn("Hello world!");
    l->error("Bye!");
}
