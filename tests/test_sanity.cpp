

#include "ctsegmentator/ctsegmentator.hpp"

#include <chrono>
#include <iostream>
#include <string>

int main()
{
    torch::Tensor tensor = torch::rand({ 2, 3 });
    std::cout << tensor << std::endl;
    ctsegmentator::Segmentator s;
    std::size_t N = 256 * 256 * 48;
    std::vector<double> im(N, 0);
    std::vector<std::uint8_t> org(N, 0);
    std::array<std::size_t, 3> shape = { 256, 256, 48 };

    for (std::size_t i = 0; i < im.size(); ++i)
        im[i] = i;

    auto jobs = s.segmentJobs(im, org, shape);
    s.useCUDA(true);
    for (const auto& job : jobs) {
        std::cout << "start ";
        auto start = std::chrono::steady_clock::now();
        bool success = s.segment(job, im, org, shape);
        auto time = std::chrono::steady_clock::now() - start;
        std::cout << success << " done ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(time).count() << std::endl;
    }

    return EXIT_SUCCESS;
    // return EXIT_FAILURE;
}
