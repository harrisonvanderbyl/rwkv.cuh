#ifndef SAMPLE_H
#define SAMPLE_H
#include <cmath>
#include <algorithm>
#include <vector>

size_t ALEN = pow(2,16);

void softmax(float* logits)
{
    float sum = 0.0;
    for (size_t i = 0; i < ALEN; i++)
    {
        sum += exp(logits[i]);
    }
    for (size_t i = 0; i < ALEN; i++)
    {
        logits[i] = exp(logits[i])/sum;
    }
};

// v $ b = log(exp(v) + exp(b))

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


size_t dart(float* logits, double _temp = 1.0)
{
    double max = double(*std::max_element(logits, logits+ALEN));
    double min = double(*std::min_element(logits, logits+ALEN));
    // create a file
    auto dart = (double(rand()) / RAND_MAX);
    dart = pow(dart, 1-pow(dart, _temp * pow(dart,10)));
   

    dart = min + dart * (max - min);
   

    
    
    auto out = std::min_element(logits, logits+ALEN, [dart](const float& lhs, const float& rhs) {
        return std::abs(lhs - dart) < std::abs(rhs - dart);
    });

    return std::distance(logits, out);
};


#endif // SAMPLE_H
