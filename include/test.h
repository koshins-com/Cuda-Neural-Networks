#ifndef TEST_H
#define TEST_H

#include <curand.h>

bool test_linear(curandGenerator_t generator);

bool test_relu(curandGenerator_t generator);

bool test_softmax(curandGenerator_t generator);

bool test_cross_entropy(curandGenerator_t generator);

bool test_cross_entropy_softmax_back(curandGenerator_t generator);

bool test_relu_back(curandGenerator_t generator);

bool test_linear_update(curandGenerator_t generator);

#endif
