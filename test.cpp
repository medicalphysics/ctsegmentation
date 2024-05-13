

#include <iostream>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>

bool load_model(const std::string &model_name)
{

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_name);
    }
    catch (const c10::Error &e)
    {
        std::cout << e.what() << std::endl;
        std::cerr << "error loading the model\n";
        return false;
    }
    return true;
}

int main()
{
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    std::string modelname = "test.pt";
    if (load_model(modelname))
        return EXIT_SUCCESS;
    return EXIT_FAILURE;
}
