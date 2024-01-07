#ifndef SAMPLE_H
#define SAMPLE_H
#include <cmath>
#include <algorithm>
#include <vector>

size_t ALEN = pow(2,16);

size_t typical(float* logits, double _temp = 0.9, double _tau = 0.9)
{
    _tau = pow(_tau, 1.0/8.0);
    double max = double(*std::max_element(logits, logits+ALEN));
    double min = double(*std::min_element(logits, logits+ALEN));
    double range = max - min;
    // move all elements to positive
    for (size_t i = 0; i < ALEN; i++)
    {
        logits[i] -= min;
        logits[i] /= range;
    }
    // apply temperature
    for (size_t i = 0; i < ALEN; i++)
    {
        logits[i] = pow(logits[i], 1.0/_temp);
    }

    // get random number between tau and 1
    double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    r = r * (1.0 - _tau) + _tau;
    
    // get the index of the element that is closest to r
    size_t out = 0;
    double min_diff = 1.0;
    for (size_t i = 0; i < ALEN; i++)
    {
        double diff = (double(logits[i]) - r);
        if (diff < 0)
        {
            diff *= -1;
        }
        if (diff < min_diff)
        {
            min_diff = diff;
            out = i;
        }
    }

    return out;
};


#endif // SAMPLE_H