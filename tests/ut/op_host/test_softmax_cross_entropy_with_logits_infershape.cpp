#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class SoftmaxCrossEntropyWithLogitsInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SoftmaxCrossEntropyWithLogitsInfershapeTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "SoftmaxCrossEntropyWithLogitsInfershapeTest TearDown" << std::endl;
    }
};

// float32, [4, 10]
TEST_F(SoftmaxCrossEntropyWithLogitsInfershapeTest, test_case_float32_2d)
{
    gert::StorageShape features_shape({4, 10}, {4, 10});
    gert::StorageShape labels_shape({4, 10}, {4, 10});
    gert::StorageShape loss_shape({}, {});
    gert::StorageShape backprop_shape({}, {});

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1})
        .InputShapes({&features_shape, &labels_shape})
        .OutputShapes({&loss_shape, &backprop_shape})
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto infer_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits")->infer_shape;
    ASSERT_NE(infer_func, nullptr);
    ASSERT_EQ(infer_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    // 验证loss shape: [4]
    auto* out_loss = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(out_loss, nullptr);
    EXPECT_EQ(out_loss->GetDimNum(), 1);
    EXPECT_EQ(out_loss->GetDim(0), 4);

    // 验证backprop shape: [4, 10]
    auto* out_backprop = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_NE(out_backprop, nullptr);
    EXPECT_EQ(out_backprop->GetDimNum(), 2);
    EXPECT_EQ(out_backprop->GetDim(0), 4);
    EXPECT_EQ(out_backprop->GetDim(1), 10);
}

// float16, [8, 100]
TEST_F(SoftmaxCrossEntropyWithLogitsInfershapeTest, test_case_float16_2d)
{
    gert::StorageShape features_shape({8, 100}, {8, 100});
    gert::StorageShape labels_shape({8, 100}, {8, 100});
    gert::StorageShape loss_shape({}, {});
    gert::StorageShape backprop_shape({}, {});

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1})
        .InputShapes({&features_shape, &labels_shape})
        .OutputShapes({&loss_shape, &backprop_shape})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto infer_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits")->infer_shape;
    ASSERT_NE(infer_func, nullptr);
    ASSERT_EQ(infer_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto* out_loss = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    EXPECT_EQ(out_loss->GetDim(0), 8);

    auto* out_backprop = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    EXPECT_EQ(out_backprop->GetDim(0), 8);
    EXPECT_EQ(out_backprop->GetDim(1), 100);
}

// bf16, [2, 50]
TEST_F(SoftmaxCrossEntropyWithLogitsInfershapeTest, test_case_bf16_2d)
{
    gert::StorageShape features_shape({2, 50}, {2, 50});
    gert::StorageShape labels_shape({2, 50}, {2, 50});
    gert::StorageShape loss_shape({}, {});
    gert::StorageShape backprop_shape({}, {});

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 2)
        .IrInstanceNum({1, 1})
        .InputShapes({&features_shape, &labels_shape})
        .OutputShapes({&loss_shape, &backprop_shape})
        .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();

    auto infer_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits")->infer_shape;
    ASSERT_NE(infer_func, nullptr);
    ASSERT_EQ(infer_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto* out_loss = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    EXPECT_EQ(out_loss->GetDim(0), 2);

    auto* out_backprop = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    EXPECT_EQ(out_backprop->GetDim(0), 2);
    EXPECT_EQ(out_backprop->GetDim(1), 50);
}