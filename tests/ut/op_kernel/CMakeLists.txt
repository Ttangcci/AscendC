if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(softmax_cross_entropy_with_logits "ascend910B1" "-DDTYPE_X=float -DDTYPE_Y=float")
endif()