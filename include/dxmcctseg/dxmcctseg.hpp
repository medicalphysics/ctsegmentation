#include <span>
#include <expected>
#include <torch/script.h>
#include <torch/torch.h>

namespace dxmcctseg {

class Segmentator {
public:
    Segmentator() = delete;

protected:
std::expected<torch::jit::script::Module, bool> loadModel(int part=0){
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_name);
    } catch (const c10::Error& e) {
        std::cout << e.what() << std::endl;
        std::cerr << "error loading the model\n";
        return false;
    }
    return true;
}

private:
    torch::jit::script::Module m_model;
}

bool
load_model(const std::string& model_name)
{

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_name);
    } catch (const c10::Error& e) {
        std::cout << e.what() << std::endl;
        std::cerr << "error loading the model\n";
        return false;
    }
    return true;
}
}
