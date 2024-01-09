#ifndef UINT8THREADING_H
#define UINT8THREADING_H
#include <atomic>
#include <thread>
#include <iostream>
// include threads

#include "tensor/tensor.h"
#include "tensor/operators/matmul/cpu.h"


// create 8 threads
// each thread does 1/8 of the work
// first, lets create a global variable to hold the threads


std::vector<std::thread *> threads;


enum JOBTYPE
{
    MATMUL,
    MATMULFP,
    RWKV_ATT
};

// create a function that will be called by the thread
struct MatMulJob
{
    const uint8_t *A = nullptr;
    const void *B;
    const void *C;
    const void *Ao;
    const void *Ar;
    const void *Bt;
    const void *Ct;
    size_t bbt;
    size_t ii;
    size_t INSHAPE;
    size_t OUTSHAPE;
    JOBTYPE type = MATMUL;
    const void *ex = nullptr;
    size_t H = 0;
    size_t hh = 0;
    TENSORTYPE dtype = TENSORTYPE::kFLOAT_32;

    MatMulJob(
        uint8_t *A = nullptr,
        void *B = nullptr,
        void *C = nullptr,
        void *Ao = nullptr,
        void *Ar = nullptr,
        void *Bt = nullptr,
        void *Ct = nullptr,
         size_t bbt = 1,
         size_t ii = 0,
         size_t INSHAPE = 0,
         size_t OUTSHAPE = 0,
         JOBTYPE type = MATMUL,
        void *ex = nullptr,
         size_t H = 0,
         size_t hh = 0,
         TENSORTYPE dtype = TENSORTYPE::kFLOAT_32
    ){
        this->A = A;
        this->B = B;
        this->C = C;
        this->Ao = Ao;
        this->Ar = Ar;
        this->Bt = Bt;
        this->Ct = Ct;
        this->bbt = bbt;
        this->ii = ii;
        this->INSHAPE = INSHAPE;
        this->OUTSHAPE = OUTSHAPE;
        this->type = type;
        this->ex = ex;
        this->H = H;
        this->hh = hh;
        this->dtype = dtype;

    }

    

    // if convert to size_t, then it will be its own address
    operator size_t() const
    {
        return (size_t)this;
    }
};

// make compatible with compiler
std::vector<std::atomic<MatMulJob*>*> jobs;

void dopartialfp(MatMulJob job);
void dopartial(MatMulJob job);

void dopartialwkv5att(MatMulJob job);

void listenfunc(std::atomic<MatMulJob*> *jobs1, std::atomic<MatMulJob*> *jobs2)
{
    // wait for all jobs to be done
    while (true)
    {
        // check if all jobs are done

        // get last job
        const auto currenta = jobs1->load();
        if (currenta != 0)
        {
            const auto current = *(MatMulJob *)currenta;

            if (current.type == JOBTYPE::RWKV_ATT)
            {
                dopartialwkv5att(current);
            }
            else
            {
                if(current.type == JOBTYPE::MATMUL){
                    dopartial(current);
                }
                else{
                    dopartialfp(current);
                }
            }

            jobs1->store(nullptr);
        }
        const auto current2 = jobs2->load();
        if (current2 != 0)
        {
            const auto current = *(MatMulJob *)current2;
            if(current.type == JOBTYPE::MATMUL){
                dopartial(current);
            }
            else{
                dopartialfp(current);
            }
            jobs2->store(nullptr);
        }
    }
}


bool started = false;
size_t ThreadCount = 8;
void startWorkers(size_t threadsNum = 8)
{

    // start the threads
    if (started)
    {
        return;
    }
    ThreadCount = threadsNum;
    started = true;

    for (size_t i = 0; i < ThreadCount; i++)
    {
        auto job1 = new std::atomic<MatMulJob *>(nullptr);
        auto job2 = new std::atomic<MatMulJob *>(nullptr);
        jobs.push_back(job1);
        jobs.push_back(job2);
        threads.push_back(new std::thread(listenfunc, job1, job2));
    }
}

void matmul8_cpu_kernal(uint8_t* A, void* B, void* C, void* Ao, void* Ar, size_t BBT, size_t INSHAPE, size_t OUTSHAPE){  


    for (size_t xc = 0; xc < 8; xc+=ThreadCount){
    for (size_t i = 0; i < ThreadCount; i+=1)
    {
        auto job1 = new MatMulJob{A, B, C, Ao, Ar, B, C, BBT, ((i+xc)*2)*16, INSHAPE, OUTSHAPE};
        auto job2 = new MatMulJob{A, B, C, Ao, Ar, B, C, BBT, ((i+xc)*2)*16+16, INSHAPE, OUTSHAPE};
        jobs[i*2]->store(job1);
        jobs[i*2+1]->store(job2);
    }


    while (true)
    {
        // check if all jobs are done
        bool done = true;
        for (size_t i = 0; i < ThreadCount; i++)
        {
            if (jobs[i*2]->load() != nullptr || jobs[i*2+1]->load() != nullptr)
            {
                done = false;
                break;
            }
        }
        if (done)
        {
            break;
        }
    }
    }
}

void matmul_cpu_kernal(void* A, void* B, void* C, size_t BBT, size_t INSHAPE, size_t OUTSHAPE, TENSORTYPE dtype){
    

    // auto job10job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 0, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    

    for (size_t xc = 0; xc < 8; xc+=ThreadCount)
    {
       

    for (size_t i = 0; i < ThreadCount; i+=1)
    {
        auto job1 = new MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, ((xc+i)*2)*16, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
        auto job2 = new MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, ((xc+i)*2)*16+16, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
        jobs[i*2]->store(job1);
        jobs[i*2+1]->store(job2);
    }

    // wait for all jobs to be done
    while (true)
    {
        // check if all jobs are done
        bool done = true;
        for (size_t i = 0; i < ThreadCount; i++)
        {
            if (jobs[i*2]->load() != nullptr || jobs[i*2+1]->load() != nullptr)
            {
                done = false;
                break;
            }
            
        }
        if (done)
        {
            break;
        }
    }
    }
}

void wkv5_cpu_kernel(void* kk, void* vv, void* ww, void* uu, void* rr, void* ss, void* out, size_t T, size_t B, size_t C, size_t H, TENSORTYPE dtype){
    

        // #pragma omp parallel for collapse(2) schedule(guided, 64) shared(kk, vv, ww, uu, rr, ss, out)
        for (uint32_t bb = 0; bb < B; bb++)
        {
            // heads are divisable by 8 I think
            for (uint32_t hh = 0; hh < H; hh += 8)
            {
                // auto job1 = MatMulJob{nullptr, out, ww, kk, vv, uu, rr, T, B, C / H, bb, JOBTYPE::RWKV_ATT, ss, H, hh + 0, dtype};

                // jobs10 = (size_t)&job1;

                for (size_t xc = 0; xc < 8; xc+=ThreadCount){
                
                   

                for (size_t i = 0; i < ThreadCount; i+=1)
                {
                    auto job1 = new MatMulJob{nullptr, out, ww, kk, vv, uu, rr, T, B, C / H, bb, JOBTYPE::RWKV_ATT, ss, H, hh + (i + xc), dtype};
                    jobs[i*2]->store(job1);
                    
                }

                // wait for all jobs to be done
                while (true)
                {
                    // check if all jobs are done
                    bool done = true;
                    for (size_t i = 0; i < ThreadCount; i++)
                    {
                        if (jobs[i*2]->load() != nullptr)
                        {
                            done = false;
                            break;
                        }
                    }
                    if (done)
                    {
                        break;
                    }
                }
                
            }}
        }
}

#endif // UINT8THREADING_H