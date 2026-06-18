#include "test.h"
#include "cucall.h"
#include <stdio.h>
#include <time.h>
#include <curand.h>

void test()
{
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, time(0)));
    printf("mul_acc:\n");
    printf(test_linear(generator)? "\tSuccess": "\tFailure");
    printf("\n");
    printf("relu:\n");
    printf(test_relu(generator)? "\tSuccess": "\tFailure");
    printf("\n");
    printf("softmax:\n");
    printf(test_softmax(generator)? "\tSuccess": "\tFailure");
    printf("\n");
    printf("loss:\n");
    printf(test_cross_entropy(generator)? "\tSuccess": "\tFailure");
    printf("\n");
    printf("loss-softmax back:\n");
    printf(test_cross_entropy_softmax_back(generator)? "\tSuccess": "\tFailure");
    printf("\n");
    printf("relu back:\n");
    printf(test_relu_back(generator)? "\tSuccess": "\tFailure");
    printf("\n");
    printf("linear update:\n");
    printf(test_linear_update(generator)? "\tSuccess": "\tFailure");
    printf("\n");
}

int main()
{
    test();
}
