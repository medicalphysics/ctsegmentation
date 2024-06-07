

#include <iostream>
#include <string>

#include "ctsegmentator/ctsegmentator.hpp"

int main()
{
    torch::Tensor tensor = torch::rand({ 2, 3 });
    std::cout << tensor << std::endl;
    ctsegmentator::Segmentator s;
    std::size_t N = 256 * 256 * 48;
    std::vector<double> im(N, 0);
    std::vector<std::uint8_t> org(N, 0);
    std::array<std::size_t, 3> shape = { 256, 256, 48 };

    if (s.segment(im, org, shape))
        return EXIT_SUCCESS;
    return EXIT_FAILURE;
}
