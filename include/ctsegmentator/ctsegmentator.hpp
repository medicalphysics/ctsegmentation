#include <array>
#include <atomic>
#include <span>
#include <string>

#include <torch/script.h>
#include <torch/torch.h>

namespace ctsegmentator {

class Segmentator {
public:
    static constexpr std::array<float, 3> spacing()
    {
        return { 1.5f, 1.5f, 1.5f };
    }

    const std::array<std::int64_t, 2>& modelShape() const
    {
        return m_model_shape;
    }
    void setModelShape(std::int64_t x, std::int64_t y)
    {
        m_model_shape[0] = std::max(x, std::int64_t { 256 });
        m_model_shape[1] = std::max(y, std::int64_t { 256 });
    }

    static constexpr std::int64_t batchSize()
    {
        return 16;
    }
    static constexpr std::int64_t modelSize()
    {
        return 16;
    }

    std::array<int, 2> progress() const
    {

        auto t_ref = std::atomic_ref(m_tasks);
        auto n_ref = std::atomic_ref(m_total_task);

        std::array<int, 2> p = {
            t_ref.load(),
            n_ref.load()
        };
        return p;
    }

    bool segment(std::span<const double> ct_raw, std::span<std::uint8_t> org_out, const std::array<std::size_t, 3>& dataShape)
    {
        const std::array<std::int64_t, 3> sh = {
            static_cast<std::int64_t>(dataShape[0]),
            static_cast<std::int64_t>(dataShape[1]),
            static_cast<std::int64_t>(dataShape[2])
        };

        const auto ct_prep = transformCTData<double>(ct_raw);

        const auto indices = tensorIndices(sh);
        m_total_task = indices.size() * 4;
        bool success = segmentPart(ct_prep, org_out, sh, indices, 1);
        success = success && segmentPart(ct_prep, org_out, sh, indices, 2);
        success = success && segmentPart(ct_prep, org_out, sh, indices, 3);
        success = success && segmentPart(ct_prep, org_out, sh, indices, 4);
        return success;
    }

    bool segment(std::span<const float> ct_raw, std::span<std::uint8_t> org_out, const std::array<std::size_t, 3>& dataShape)
    {
        const std::array<std::int64_t, 3> sh = {
            static_cast<std::int64_t>(dataShape[0]),
            static_cast<std::int64_t>(dataShape[1]),
            static_cast<std::int64_t>(dataShape[2])
        };

        const auto ct_prep = transformCTData<float>(ct_raw);

        const auto indices = tensorIndices(sh);
        m_total_task = indices.size() * 4;
        bool success = segmentPart(ct_prep, org_out, sh, indices, 1);
        success = success && segmentPart(ct_prep, org_out, sh, indices, 2);
        success = success && segmentPart(ct_prep, org_out, sh, indices, 3);
        success = success && segmentPart(ct_prep, org_out, sh, indices, 4);
        return success;
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

    template <std::floating_point T>
    static constexpr std::vector<float> transformCTData(std::span<const T> arr)
    {
        std::vector<float> c(arr.size());
        std::transform(arr.cbegin(), arr.cend(), c.begin(), [](const auto av) { return (static_cast<float>(av) + 1024) / 2048; });
        return c;
    }

    bool segmentPart(std::span<const float> ct_in, std::span<std::uint8_t> org_out, const std::array<std::int64_t, 3>& dataShape,
        const std::vector<std::array<std::int64_t, 6>>& indices, int part = 0)
    {
        bool success = loadModel(part);
        success = success && ct_in.size() == org_out.size();
        if (!success)
            return success;

        auto progress_ref = std::atomic_ref(m_tasks);

        auto in = torch::empty({ batchSize(), 1, modelShape()[0], modelShape()[1] }, torch::dtype(torch::kFloat32));
        auto in_acc = in.accessor<float, 4>();

        torch::NoGradGuard no_grad;
        m_model.eval();

        for (const auto& startstop : indices) {
            in.fill_(0);
            for (auto z = startstop[2]; z < startstop[5]; ++z)
                for (auto y = startstop[1]; y < startstop[4]; ++y)
                    for (auto x = startstop[0]; x < startstop[3]; ++x) {
                        const auto tx = x - startstop[0];
                        const auto ty = y - startstop[1];
                        const auto tz = z - startstop[2];
                        const auto ctIdx = z * dataShape[0] * dataShape[1] + y * dataShape[0] + x;

                        in_acc[tz][0][ty][tx] = ct_in[ctIdx];
                        // in.index_put_({ tz, 0, ty, tx }, ct_in[ctIdx]);
                    }
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(in);
            auto out = m_model.forward(inputs).toTensor();
            auto out_acc = out.accessor<float, 4>();
            for (auto z = startstop[2]; z < startstop[5]; ++z)
                for (std::int64_t c = 0; c < modelSize(); ++c)
                    for (auto y = startstop[1]; y < startstop[4]; ++y)
                        for (auto x = startstop[0]; x < startstop[3]; ++x) {
                            const auto tx = x - startstop[0];
                            const auto ty = y - startstop[1];
                            const auto tz = z - startstop[2];
                            // if (out.index({ tz, c, ty, tx }).item<float>() > 0.5f) {
                            if (out_acc[tz][c][ty][tx] > 0.5f) {
                                const auto ctIdx = z * dataShape[0] * dataShape[1] + y * dataShape[0] + x;
                                org_out[ctIdx] = static_cast<std::uint8_t>(c + part * modelSize());
                            }
                        }
            progress_ref.fetch_add(1);
        }
        return true;
    }

    bool loadModel(int part = 0)
    {
        if (part < 0 || part > 3)
            return false;
        const std::array<std::string, 4> names = { "freezed_model1.pt", "freezed_model2.pt", "freezed_model3.pt", "freezed_model4.pt" };
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            m_model = torch::jit::load(names[part]);
        } catch (const c10::Error& e) {
            // std::cout << e.what() << std::endl;
            // std::cerr << "error loading the model\n";
            return false;
        }
        return true;
    }

private:
    torch::jit::script::Module m_model;
    std::array<std::int64_t, 2> m_model_shape = { 256, 256 };
    int m_tasks = 0;
    int m_total_task = 0;
};

}
