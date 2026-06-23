#include <array>
#include <vector>
#include <string>
#include <iostream>
#include "gtest/gtest.h"
#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include <string.h>
#endif
#include "../../../op_kernel/softmax_cross_entropy_with_logits.cpp"
#include "../../../op_kernel/softmax_cross_entropy_with_logits_tiling_data.h"

using namespace std;

class SoftmaxCrossEntropyWithLogitsKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "SoftmaxCrossEntropyWithLogitsKernelTest SetUp" << endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./scewl_data/");
    }
    static void TearDownTestCase()
    {
        cout << "SoftmaxCrossEntropyWithLogitsKernelTest TearDown" << endl;
    }
private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string SoftmaxCrossEntropyWithLogitsKernelTest::rootPath = "../../../../";
const std::string SoftmaxCrossEntropyWithLogitsKernelTest::dataPath =
    rootPath + "experimental/activation/softmax_cross_entropy_with_logits/tests/ut/op_kernel/scewl_data";

// float32, shape [4, 10]
TEST_F(SoftmaxCrossEntropyWithLogitsKernelTest, test_case_float32)
{
    system("cd ./scewl_data/ && python3 gen_data.py '(4, 10)' 'float32'");

    const uint64_t batch      = 4;
    const uint64_t numClasses = 10;
    const uint64_t elemSize   = sizeof(float);

    uint64_t featuresSize = batch * numClasses * elemSize;
    uint64_t lossSize     = batch * elemSize;
    uint64_t backpropSize = batch * numClasses * elemSize;

    uint8_t* features  = (uint8_t*)AscendC::GmAlloc(featuresSize);
    uint8_t* labels    = (uint8_t*)AscendC::GmAlloc(featuresSize);
    uint8_t* loss      = (uint8_t*)AscendC::GmAlloc(lossSize);
    uint8_t* backprop  = (uint8_t*)AscendC::GmAlloc(backpropSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tilingBuf = (uint8_t*)AscendC::GmAlloc(sizeof(SoftmaxCrossEntropyWithLogitsTilingData));

    ReadFile("./scewl_data/float32_features.bin", featuresSize, features, featuresSize);
    ReadFile("./scewl_data/float32_labels.bin",   featuresSize, labels,   featuresSize);

    auto* tiling        = reinterpret_cast<SoftmaxCrossEntropyWithLogitsTilingData*>(tilingBuf);
    tiling->batchSize   = batch;
    tiling->numClasses  = numClasses;
    tiling->blockLength = batch;
    tiling->tileNum     = 1;
    tiling->tileLength  = batch;

    auto kernel = [](GM_ADDR features, GM_ADDR labels, GM_ADDR loss,
                     GM_ADDR backprop, GM_ADDR workspace, GM_ADDR tiling) {
        ::softmax_cross_entropy_with_logits<0>(features, labels, loss, backprop, workspace, tiling);
    };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1, features, labels, loss, backprop, workspace, tilingBuf);

    WriteFile("./scewl_data/float32_loss_actual.bin",    loss,    lossSize);
    WriteFile("./scewl_data/float32_backprop_actual.bin", backprop, backpropSize);

    AscendC::GmFree(features);
    AscendC::GmFree(labels);
    AscendC::GmFree(loss);
    AscendC::GmFree(backprop);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingBuf);

    int ret = system("cd ./scewl_data/ && python3 compare_data.py 'float32'");
    EXPECT_EQ(ret, 0);
}

// float16, shape [8, 20]
TEST_F(SoftmaxCrossEntropyWithLogitsKernelTest, test_case_float16)
{
    system("cd ./scewl_data/ && python3 gen_data.py '(8, 20)' 'float16'");

    const uint64_t batch      = 8;
    const uint64_t numClasses = 20;
    const uint64_t elemSize   = sizeof(uint16_t);

    uint64_t featuresSize = batch * numClasses * elemSize;
    uint64_t lossSize     = batch * elemSize;
    uint64_t backpropSize = batch * numClasses * elemSize;

    uint8_t* features  = (uint8_t*)AscendC::GmAlloc(featuresSize);
    uint8_t* labels    = (uint8_t*)AscendC::GmAlloc(featuresSize);
    uint8_t* loss      = (uint8_t*)AscendC::GmAlloc(lossSize);
    uint8_t* backprop  = (uint8_t*)AscendC::GmAlloc(backpropSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tilingBuf = (uint8_t*)AscendC::GmAlloc(sizeof(SoftmaxCrossEntropyWithLogitsTilingData));

    ReadFile("./scewl_data/float16_features.bin", featuresSize, features, featuresSize);
    ReadFile("./scewl_data/float16_labels.bin",   featuresSize, labels,   featuresSize);

    auto* tiling        = reinterpret_cast<SoftmaxCrossEntropyWithLogitsTilingData*>(tilingBuf);
    tiling->batchSize   = batch;
    tiling->numClasses  = numClasses;
    tiling->blockLength = batch;
    tiling->tileNum     = 1;
    tiling->tileLength  = batch;

    auto kernel = [](GM_ADDR features, GM_ADDR labels, GM_ADDR loss,
                     GM_ADDR backprop, GM_ADDR workspace, GM_ADDR tiling) {
        ::softmax_cross_entropy_with_logits<1>(features, labels, loss, backprop, workspace, tiling);
    };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1, features, labels, loss, backprop, workspace, tilingBuf);

    WriteFile("./scewl_data/float16_loss_actual.bin",     loss,    lossSize);
    WriteFile("./scewl_data/float16_backprop_actual.bin", backprop, backpropSize);

    AscendC::GmFree(features);
    AscendC::GmFree(labels);
    AscendC::GmFree(loss);
    AscendC::GmFree(backprop);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingBuf);

    int ret = system("cd ./scewl_data/ && python3 compare_data.py 'float16'");
    EXPECT_EQ(ret, 0);
}
// bf16, shape [4, 10]
TEST_F(SoftmaxCrossEntropyWithLogitsKernelTest, test_case_bf16)
{
    system("cd ./scewl_data/ && python3 gen_data.py '(4, 10)' 'bfloat16'");

    const uint64_t batch      = 4;
    const uint64_t numClasses = 10;
    const uint64_t elemSize   = sizeof(uint16_t);  // bf16是2字节

    uint64_t featuresSize = batch * numClasses * elemSize;
    uint64_t lossSize     = batch * elemSize;
    uint64_t backpropSize = batch * numClasses * elemSize;

    uint8_t* features  = (uint8_t*)AscendC::GmAlloc(featuresSize);
    uint8_t* labels    = (uint8_t*)AscendC::GmAlloc(featuresSize);
    uint8_t* loss      = (uint8_t*)AscendC::GmAlloc(lossSize);
    uint8_t* backprop  = (uint8_t*)AscendC::GmAlloc(backpropSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tilingBuf = (uint8_t*)AscendC::GmAlloc(sizeof(SoftmaxCrossEntropyWithLogitsTilingData));

    ReadFile("./scewl_data/bfloat16_features.bin", featuresSize, features, featuresSize);
    ReadFile("./scewl_data/bfloat16_labels.bin",   featuresSize, labels,   featuresSize);

    auto* tiling        = reinterpret_cast<SoftmaxCrossEntropyWithLogitsTilingData*>(tilingBuf);
    tiling->batchSize   = batch;
    tiling->numClasses  = numClasses;
    tiling->blockLength = batch;
    tiling->tileNum     = 1;
    tiling->tileLength  = batch;

    auto kernel = [](GM_ADDR features, GM_ADDR labels, GM_ADDR loss,
                     GM_ADDR backprop, GM_ADDR workspace, GM_ADDR tiling) {
        ::softmax_cross_entropy_with_logits<2>(features, labels, loss, backprop, workspace, tiling);
    };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1, features, labels, loss, backprop, workspace, tilingBuf);

    WriteFile("./scewl_data/bfloat16_loss_actual.bin",     loss,    lossSize);
    WriteFile("./scewl_data/bfloat16_backprop_actual.bin", backprop, backpropSize);

    AscendC::GmFree(features);
    AscendC::GmFree(labels);
    AscendC::GmFree(loss);
    AscendC::GmFree(backprop);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingBuf);

    int ret = system("cd ./scewl_data/ && python3 compare_data.py 'bfloat16'");
    EXPECT_EQ(ret, 0);
}