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

size_t typical(float* logits, double _temp = 3.0, double _tau = 0.6)
{
    softmax(logits);
    
    float sorted_probs[ALEN];
    std::copy(logits, logits+ALEN, sorted_probs);
    std::sort(sorted_probs, sorted_probs+ALEN, std::greater<float>());
    float cumulative_probs[ALEN];
    cumulative_probs[0] = sorted_probs[0];
    for (size_t i = 1; i < ALEN; i++)
    {
        cumulative_probs[i] = cumulative_probs[i-1] + sorted_probs[i];
    }
    float cutoff = sorted_probs[0];
    for (size_t i = 0; i < ALEN; i++)
    {
        if (cumulative_probs[i] > _tau)
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
    if (_temp != 1.0)
    {
        for (size_t i = 0; i < ALEN; i++)
        {
            logits[i] = pow(logits[i], 1.0/_temp);
        }
    }
    float sum = 0;
    for (size_t i = 0; i < ALEN; i++)
    {
        sum += logits[i];
    }
    for (size_t i = 0; i < ALEN; i++)
    {
        logits[i] /= sum;
    }
    float r = (float)rand() / (float)RAND_MAX;
    float cumulative = 0;
    size_t out = 0;
    for (size_t i = 0; i < ALEN; i++)
    {
        cumulative += logits[i];
        if (cumulative > r)
        {
            out = i;
            break;
        }
    }
    return out;
};


#endif // SAMPLE_H