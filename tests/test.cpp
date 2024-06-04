

#include <iostream>
#include <string>

#include "dxmcctseg/dxmcctseg.hpp"


int main()
{
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    std::string modelname = "freezed_model1.pt";
    if (dxmcctseg::load_model(modelname))
        return EXIT_SUCCESS;
    return EXIT_FAILURE;
}
