#ifndef TEST_H
#define TEST_H

#include <curand.h>

bool test_mul_acc(curandGenerator_t generator);

bool test_relu(curandGenerator_t generator);

bool test_softmax(curandGenerator_t generator);

bool test_loss(curandGenerator_t generator);

#endif
