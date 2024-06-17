
#pragma once

#include "organlist.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <array>
#include <atomic>
#include <span>
#include <string>
#include <thread>

#include <iostream>

namespace ctsegmentator {

enum class ModelPart {
    Empty,
    Model1,
    Model2,
    Model3,
    Model4
};

struct Job {
    std::array<std::int64_t, 3> start;
    std::array<std::int64_t, 3> stop;
    ModelPart part = ModelPart::Empty;
};

class Segmentator {
public:
    Segmentator()
    {
        const auto nt = static_cast<int>(std::thread::hardware_concurrency());
        torch::set_num_threads(std::max(nt, 1));
    }

    void useCUDA(bool enable)
    {
        m_useCuda = torch::cuda::is_available() && enable;
    }

    const std::array<std::int64_t, 2>& modelShape() const
    {
        return m_model_shape;
    }

    static constexpr std::int64_t batchSize()
    {
        return 32;
    }

    static constexpr std::int64_t modelSize()
    {
        return 16;
    }

    static const std::map<std::uint8_t, std::string>& organNames()
    {
        return organ_names;
    }

    std::vector<Job> segmentJobs(std::span<const double> ct_raw, std::span<std::uint8_t> org_out, const std::array<std::size_t, 3>& dataShape) const
    {
        const std::array<std::int64_t, 3> sh = {
            static_cast<std::int64_t>(dataShape[0]),
            static_cast<std::int64_t>(dataShape[1]),
            static_cast<std::int64_t>(dataShape[2])
        };

        const auto indices = tensorIndices(sh);
        const auto N = indices.size();
        std::vector<Job> jobs(N * 4);
        const std::array<ModelPart, 4> model = {
            ModelPart::Model1,
            ModelPart::Model2,
            ModelPart::Model3,
            ModelPart::Model4
        };
        for (std::size_t i = 0; i < model.size(); i++) {
            for (std::size_t j = 0; j < N; ++j) {
                auto& job = jobs[j + N * i];
                for (std::size_t k = 0; k < 3; ++k) {
                    job.start[k] = indices[j][k];
                    job.stop[k] = indices[j][k + 3];
                    job.part = model[i];
                }
            }
        }
        return jobs;
    }

    bool segment(const Job& job, std::span<const double> ct_raw, std::span<std::uint8_t> org_out, const std::array<std::size_t, 3>& dataShape)
    {
        const std::array<std::int64_t, 3> sh = {
            static_cast<std::int64_t>(dataShape[0]),
            static_cast<std::int64_t>(dataShape[1]),
            static_cast<std::int64_t>(dataShape[2])
        };

        bool success = loadModel(job.part);
        if (!success)
            return success;

        const auto modelIndex = [](const ModelPart& part) -> std::int64_t {
            switch (part) {
            case ModelPart::Model1:
                return 0;
            case ModelPart::Model2:
                return 1;
            case ModelPart::Model3:
                return 2;
            case ModelPart::Model4:
                return 3;
            default:
                return -1;
            }
        }(job.part);

        auto in = torch::empty({ batchSize(), 1, modelShape()[0], modelShape()[1] }, torch::dtype(torch::kFloat32));
        auto in_acc = in.accessor<float, 4>();

        torch::NoGradGuard no_grad;
        m_model.eval();
        in.fill_(float { -1024 });
        for (auto z = job.start[2]; z < job.stop[2]; ++z)
            for (auto y = job.start[1]; y < job.stop[1]; ++y)
                for (auto x = job.start[0]; x < job.stop[0]; ++x) {
                    const auto ty = x - job.start[0];
                    const auto tx = job.stop[1] - 1 - (y - job.start[1]);
                    const auto tz = z - job.start[2];
                    const auto ctIdx = z * dataShape[0] * dataShape[1] + y * dataShape[0] + x;

                    in_acc[tz][0][ty][tx] = static_cast<float>(ct_raw[ctIdx]);
                }
        if (m_useCuda) {
            in.cuda();
        }
        in.add_(float { 1024 });
        in.div_(float { 2048 });
        in.clamp_(float { 0 }, float { 1 });

        // saveTensor(in, "in.zip");

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(in);
        auto out = m_model.forward(inputs).toTensor().cpu();
        auto out_acc = out.accessor<float, 4>();
        for (auto z = job.start[2]; z < job.stop[2]; ++z)
            for (std::int64_t c = 1; c < modelSize(); ++c)
                for (auto y = job.start[1]; y < job.stop[1]; ++y)
                    for (auto x = job.start[0]; x < job.stop[0]; ++x) {
                        const auto ty = x - job.start[0];
                        const auto tx = job.stop[1] - 1 - (y - job.start[1]);
                        const auto tz = z - job.start[2];
                        if (out_acc[tz][c][ty][tx] > 0.5f) {
                            const auto ctIdx = z * dataShape[0] * dataShape[1] + y * dataShape[0] + x;
                            org_out[ctIdx] = static_cast<std::uint8_t>(c + modelIndex * (modelSize() - 1));
                        }
                    }
        return success;
    }

    void saveTensor(const auto& t, const std::string& fname)
    {
        auto bytes = torch::jit::pickle_save(t);
        std::ofstream fout(fname, std::ios::out | std::ios::binary);
        fout.write(bytes.data(), bytes.size());
        fout.close();
    }

protected:
    std::vector<std::array<std::int64_t, 6>> tensorIndices(const std::array<std::int64_t, 3>& dataShape) const
    {
        const auto mSh = modelShape();
        std::int64_t nx = dataShape[0] / mSh[0];
        if (nx * mSh[0] < dataShape[0])
            nx++;
        std::int64_t ny = dataShape[1] / mSh[1];
        if (ny * mSh[1] < dataShape[1])
            ny++;
        std::int64_t nz = dataShape[2] / batchSize();
        if (nz * batchSize() < dataShape[2])
            nz++;

        std::vector<std::array<std::int64_t, 6>> indices;
        indices.reserve(nx * ny * nz);

        for (std::int64_t k = 0; k < nz; k++)
            for (std::int64_t j = 0; j < ny; j++)
                for (std::int64_t i = 0; i < nx; i++) {
                    auto bIdx = k * ny * nx + j * nx + i;
                    std::array<std::int64_t, 6> startstop = {
                        i * mSh[0],
                        j * mSh[1],
                        k * batchSize(),
                        std::min((i + 1) * mSh[0], dataShape[0]),
                        std::min((j + 1) * mSh[1], dataShape[1]),
                        std::min((k + 1) * batchSize(), dataShape[2]),
                    };
                    indices.push_back(startstop);
                }
        return indices;
    }

    bool loadModel(ModelPart part = ModelPart::Empty)
    {
        if (part == ModelPart::Empty)
            return false;

        if (m_current_model != part) {
            std::string name;
            if (part == ModelPart::Model1)
                name = m_useCuda ? "freezed_cuda_model1.pt" : "freezed_model1.pt";
            else if (part == ModelPart::Model2)
                name = m_useCuda ? "freezed_cuda_model2.pt" : "freezed_model2.pt";
            else if (part == ModelPart::Model3)
                name = m_useCuda ? "freezed_cuda_model3.pt" : "freezed_model3.pt";
            else if (part == ModelPart::Model4)
                name = m_useCuda ? "freezed_cuda_model4.pt" : "freezed_model4.pt";
            try {
                // Deserialize the ScriptModule from a file using torch::jit::load().
                m_model = torch::jit::load(name);
            } catch (const c10::Error& e) {
                // std::cout << e.what() << std::endl;
                // std::cerr << "error loading the model\n";
                return false;
            }
            m_current_model = part;
        }

        return true;
    }

private:
    torch::jit::script::Module m_model;
    ModelPart m_current_model = ModelPart::Empty;
    const std::array<std::int64_t, 2> m_model_shape = { 384, 384 };
    bool m_useCuda = false;
};

}
