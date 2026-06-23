#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "../../../op_kernel/softmax_cross_entropy_with_logits_tiling_data.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class SoftmaxCrossEntropyWithLogitsTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SoftmaxCrossEntropyWithLogitsTilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "SoftmaxCrossEntropyWithLogitsTilingTest TearDown" << std::endl;
    }
};

// float32, [4, 10]
TEST_F(SoftmaxCrossEntropyWithLogitsTilingTest, test_case_float32)
{
    gert::StorageShape features_shape({4, 10}, {4, 10});
    gert::StorageShape labels_shape({4, 10}, {4, 10});
    gert::StorageShape loss_shape({4}, {4});
    gert::StorageShape backprop_shape({4, 10}, {4, 10});

    string compile_info_string = R"({
        "hardware_info": {
            "UB_SIZE": 196608,
            "CORE_NUM": 48
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct SoftmaxCrossEntropyWithLogitsCompileInfo {} compile_info;

    auto kernel_holder = gert::KernelRunContextFaker()
        .KernelIONum(1, 1)
        .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
        .Outputs({&compile_info})
        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
        .SetOpType("SoftmaxCrossEntropyWithLogits")
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1})
        .InputShapes({&features_shape, &labels_shape})
        .OutputShapes({&loss_shape, &backprop_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .TilingData(param.get())
        .Workspace(ws_size)
        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits")->tiling;
    ASSERT_NE(tiling_func, nullptr);
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    // 验证tiling数据
    //auto* tiling_data = reinterpret_cast<SoftmaxCrossEntropyWithLogitsTilingData*>(param->GetData());
    auto* tiling_data = reinterpret_cast<SoftmaxCrossEntropyWithLogitsTilingData*>(tiling_context->GetRawTilingData()->GetData());
    EXPECT_EQ(tiling_data->batchSize, 4);
    EXPECT_EQ(tiling_data->numClasses, 10);
}

// float16, [8, 100]
TEST_F(SoftmaxCrossEntropyWithLogitsTilingTest, test_case_float16)
{
    gert::StorageShape features_shape({8, 100}, {8, 100});
    gert::StorageShape labels_shape({8, 100}, {8, 100});
    gert::StorageShape loss_shape({8}, {8});
    gert::StorageShape backprop_shape({8, 100}, {8, 100});

    string compile_info_string = R"({
        "hardware_info": {
            "UB_SIZE": 196608,
            "CORE_NUM": 48
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct SoftmaxCrossEntropyWithLogitsCompileInfo {} compile_info;

    auto kernel_holder = gert::KernelRunContextFaker()
        .KernelIONum(1, 1)
        .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
        .Outputs({&compile_info})
        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    auto holder = gert::TilingContextFaker()
        .SetOpType("SoftmaxCrossEntropyWithLogits")
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1})
        .InputShapes({&features_shape, &labels_shape})
        .OutputShapes({&loss_shape, &backprop_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .TilingData(param.get())
        .Workspace(ws_size)
        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits")->tiling;
    ASSERT_NE(tiling_func, nullptr);
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    //auto* tiling_data = reinterpret_cast<SoftmaxCrossEntropyWithLogitsTilingData*>(param->GetData());
    auto* tiling_data = reinterpret_cast<SoftmaxCrossEntropyWithLogitsTilingData*>(tiling_context->GetRawTilingData()->GetData());
    EXPECT_EQ(tiling_data->batchSize, 8);
    EXPECT_EQ(tiling_data->numClasses, 100);
}
// bf16, shape [2, 50]
TEST_F(SoftmaxCrossEntropyWithLogitsTilingTest, test_case_bf16)
{
    gert::StorageShape features_shape({2, 50}, {2, 50});
    gert::StorageShape labels_shape({2, 50}, {2, 50});
    gert::StorageShape loss_shape({2}, {2});
    gert::StorageShape backprop_shape({2, 50}, {2, 50});

    string compile_info_string = R"({
        "hardware_info": {
            "UB_SIZE": 196608,
            "CORE_NUM": 48
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct SoftmaxCrossEntropyWithLogitsCompileInfo {} compile_info;

    auto kernel_holder = gert::KernelRunContextFaker()
        .KernelIONum(1, 1)
        .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
        .Outputs({&compile_info})
        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    auto holder = gert::TilingContextFaker()
        .SetOpType("SoftmaxCrossEntropyWithLogits")
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1})
        .InputShapes({&features_shape, &labels_shape})
        .OutputShapes({&loss_shape, &backprop_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .TilingData(param.get())
        .Workspace(ws_size)
        .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits")->tiling;
    ASSERT_NE(tiling_func, nullptr);
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto* tiling_data = reinterpret_cast<SoftmaxCrossEntropyWithLogitsTilingData*>(
        tiling_context->GetRawTilingData()->GetData());
    EXPECT_EQ(tiling_data->batchSize, 2);
    EXPECT_EQ(tiling_data->numClasses, 50);
}