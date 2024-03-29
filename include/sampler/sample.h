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

/*
def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out.float().cpu(), dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out*/

size_t typical(float* logits, double _temp = 1.33, double _tau = 0.0)
{
    
    softmax(logits);
    float* sorted_probs = new float[ALEN];
    std::copy(logits, logits+ALEN, sorted_probs);
    // sort from smallest to largest
    std::sort(sorted_probs, sorted_probs+ALEN, [](const float& lhs, const float& rhs) {
        return lhs < rhs;
    });
    float* cumulative_probs = new float[ALEN];
    cumulative_probs[0] = sorted_probs[0];
    for (size_t i = 1; i < ALEN; i++)
    {
        cumulative_probs[i] = cumulative_probs[i-1] + sorted_probs[i];
    }
    float cutoff = sorted_probs[0];
    auto tau = _tau*cumulative_probs[ALEN-1];
    for (size_t i = 0; i < ALEN; i++)
    {
        if (cumulative_probs[i] > tau)
        {
            cutoff = sorted_probs[i];
            break;
        }
    }
    for (size_t i = 0; i < ALEN; i++)
    {
        if (logits[i] < cutoff)
        {
            logits[i] = 0;
        }
    }

    
    float sum1 = 0.0;
    for (size_t i = 0; i < ALEN; i++)
    {
        sum1 = std::max(logits[i], sum1);
    }

    float r = ((double)rand()) / ((double)RAND_MAX);
    r = pow(r, _temp) * sum1;
    size_t out = 1;

    struct prob{
        float p;
        size_t i;
    };



    std::vector<prob> probs;

    for (size_t i = 0; i < ALEN; i++)
    {
        if (logits[i] > 0.01)
        {
            prob p;
            p.p = logits[i];
            p.i = i;
            probs.push_back(p);
        }
    }

    // sort fom smallest to largest

    std::sort(probs.begin(), probs.end(), [](const prob& lhs, const prob& rhs) {
        return lhs.p < rhs.p;
    });


    for (size_t i = 0; i < probs.size(); i++)
    {
        if (probs[i].p > r)
        {
            out = probs[i].i;
            break;
        }
    }
    
    
    return out;
};


size_t dart(float* logits, double _temp = 1.0, double _tau = 0.6)
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
