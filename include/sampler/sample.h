#ifndef SAMPLE_H
#define SAMPLE_H
#include <cmath>
#include <algorithm>
#include <vector>

size_t ALEN = pow(2,16);

void softmax(float* logits)
{
    double max = double(*std::max_element(logits, logits+ALEN));
    double sum = 0;
    for (size_t i = 0; i < ALEN; i++)
    {
        logits[i] -= max;
        logits[i] = exp(logits[i]);
        sum += logits[i];
    }
    for (size_t i = 0; i < ALEN; i++)
    {
        logits[i] /= sum;
    }
};

size_t typical(float* logits, double _temp = 3.0, double _tau = 0.6)
{
    softmax(logits);
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
    

    // get random number between tau and 1
    double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    if(abs(_temp) < 0.0001){
        _temp = 0.0001;
    }

    r = pow(r, 1.0/_temp);
    
    r = (1.0-r) * (1.0 - _tau) + _tau;
    
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