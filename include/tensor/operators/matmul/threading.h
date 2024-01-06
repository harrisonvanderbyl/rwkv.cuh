#ifndef UINT8THREADING_H
#define UINT8THREADING_H
#include <atomic>
#include <thread>
#include <iostream>
// include threads

#include "tensor/tensor.h"
#include "tensor/operators/matmul/cpu.h"


size_t ZeroLong = size_t(0);
// create 8 threads
// each thread does 1/8 of the work
// first, lets create a global variable to hold the threads
std::thread *t1;
std::thread *t2;
std::thread *t3;
std::thread *t4;
std::thread *t5;
std::thread *t6;
std::thread *t7;
std::thread *t8;

enum JOBTYPE
{
    MATMUL,
    MATMULFP,
    RWKV_ATT
};

// create a function that will be called by the thread
struct MatMulJob
{
    const u_char *A = nullptr;
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
        u_char *A = nullptr,
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
std::atomic<size_t> jobs10(0);
std::atomic<size_t> jobs11(0);
std::atomic<size_t> jobs12(0);
std::atomic<size_t> jobs13(0);

std::atomic<size_t> jobs20(0);
std::atomic<size_t> jobs21(0);
std::atomic<size_t> jobs22(0);
std::atomic<size_t> jobs23(0);

std::atomic<size_t> jobs30(0);
std::atomic<size_t> jobs31(0);
std::atomic<size_t> jobs32(0);
std::atomic<size_t> jobs33(0);

std::atomic<size_t> jobs40(0);
std::atomic<size_t> jobs41(0);
std::atomic<size_t> jobs42(0);
std::atomic<size_t> jobs43(0);

void dopartialfp(MatMulJob job);
void dopartial(MatMulJob job);

void dopartialwkv5att(MatMulJob job);

void listenfunc(std::atomic<size_t> *jobs1, std::atomic<size_t> *jobs2)
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

            jobs1->store(ZeroLong);
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
            jobs2->store(ZeroLong);
        }
    }
}


bool started = false;

void startWorkers()
{
    // start the threads
    if (started)
    {
        return;
    }
    started = true;

    std::cout << "Starting workers" << std::endl;

    t1 = new std::thread(listenfunc, &jobs10, &jobs11);
    t2 = new std::thread(listenfunc, &jobs12, &jobs13);
    t3 = new std::thread(listenfunc, &jobs20, &jobs21);
    t4 = new std::thread(listenfunc, &jobs22, &jobs23);
    t5 = new std::thread(listenfunc, &jobs30, &jobs31);
    t6 = new std::thread(listenfunc, &jobs32, &jobs33);
    t7 = new std::thread(listenfunc, &jobs40, &jobs41);
    t8 = new std::thread(listenfunc, &jobs42, &jobs43);
    std::cout << "Started workers" << std::endl;
}

void matmul8_cpu_kernal(u_char* A, void* B, void* C, void* Ao, void* Ar, size_t BBT, size_t INSHAPE, size_t OUTSHAPE){  

    startWorkers();

    auto job10job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 0, INSHAPE, OUTSHAPE};
    auto job11job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 16, INSHAPE, OUTSHAPE};
    auto job12job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 32, INSHAPE, OUTSHAPE};
    auto job13job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 48, INSHAPE, OUTSHAPE};
    auto job20job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 64, INSHAPE, OUTSHAPE};
    auto job21job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 80, INSHAPE, OUTSHAPE};
    auto job22job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 96, INSHAPE, OUTSHAPE};
    auto job23job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 112, INSHAPE, OUTSHAPE};
    auto job30job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 128, INSHAPE, OUTSHAPE};
    auto job31job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 144, INSHAPE, OUTSHAPE};
    auto job32job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 160, INSHAPE, OUTSHAPE};
    auto job33job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 176, INSHAPE, OUTSHAPE};
    auto job40job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 192, INSHAPE, OUTSHAPE};
    auto job41job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 208, INSHAPE, OUTSHAPE};
    auto job42job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 224, INSHAPE, OUTSHAPE};
    auto job43job = MatMulJob{A, B, C, Ao, Ar, B, C, BBT, 240, INSHAPE, OUTSHAPE};

    jobs10 = job10job;
    jobs11 = job11job;
    jobs12 = job12job;
    jobs13 = job13job;
    jobs20 = job20job;
    jobs21 = job21job;
    jobs22 = job22job;
    jobs23 = job23job;
    jobs30 = job30job;
    jobs31 = job31job;
    jobs32 = job32job;
    jobs33 = job33job;
    jobs40 = job40job;
    jobs41 = job41job;
    jobs42 = job42job;
    jobs43 = job43job;

    while (
        jobs10 != 0 | jobs11 != 0 | jobs12 != 0 | jobs13 != 0 |
        jobs20 != 0 | jobs21 != 0 | jobs22 != 0 | jobs23 != 0 |
        jobs30 != 0 | jobs31 != 0 | jobs32 != 0 | jobs33 != 0 |
        jobs40 != 0 | jobs41 != 0 | jobs42 != 0 | jobs43 != 0)
    {
    }
}

void matmul_cpu_kernal(void* A, void* B, void* C, size_t BBT, size_t INSHAPE, size_t OUTSHAPE, TENSORTYPE dtype){
    startWorkers();

    auto job10job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 0, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job11job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 16, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job12job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 32, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job13job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 48, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job20job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 64, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job21job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 80, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job22job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 96, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job23job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 112, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job30job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 128, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job31job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 144, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job32job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 160, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job33job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 176, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job40job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 192, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job41job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 208, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job42job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 224, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};
    auto job43job = MatMulJob{nullptr, B, C, A, nullptr, B, C, BBT, 240, INSHAPE, OUTSHAPE, JOBTYPE::MATMULFP,nullptr,0,0, dtype};

    jobs10 = job10job;
    jobs11 = job11job;
    jobs12 = job12job;
    jobs13 = job13job;
    jobs20 = job20job;
    jobs21 = job21job;
    jobs22 = job22job;
    jobs23 = job23job;
    jobs30 = job30job;
    jobs31 = job31job;
    jobs32 = job32job;
    jobs33 = job33job;
    jobs40 = job40job;
    jobs41 = job41job;
    jobs42 = job42job;
    jobs43 = job43job;

    while (
        jobs10 != 0 | jobs11 != 0 | jobs12 != 0 | jobs13 != 0 |
        jobs20 != 0 | jobs21 != 0 | jobs22 != 0 | jobs23 != 0 |
        jobs30 != 0 | jobs31 != 0 | jobs32 != 0 | jobs33 != 0 |
        jobs40 != 0 | jobs41 != 0 | jobs42 != 0 | jobs43 != 0)
    {
    }
}

void wkv5_cpu_kernel(void* kk, void* vv, void* ww, void* uu, void* rr, void* ss, void* out, size_t T, size_t B, size_t C, size_t H, TENSORTYPE dtype){
     startWorkers();

        // #pragma omp parallel for collapse(2) schedule(guided, 64) shared(kk, vv, ww, uu, rr, ss, out)
        for (uint32_t bb = 0; bb < B; bb++)
        {
            // heads are divisable by 8 I think
            for (uint32_t hh = 0; hh < H; hh += 8)
            {
                auto job1 = MatMulJob{nullptr, out, ww, kk, vv, uu, rr, T, B, C / H, bb, JOBTYPE::RWKV_ATT, ss, H, hh + 0, dtype};

                jobs10 = (size_t)&job1;
                auto job2 = MatMulJob{nullptr, out, ww, kk, vv, uu, rr, T, B, C / H, bb, JOBTYPE::RWKV_ATT, ss, H, hh + 1, dtype};

                jobs12 = (size_t)&job2;
                auto job3 = MatMulJob{nullptr, out, ww, kk, vv, uu, rr, T, B, C / H, bb, JOBTYPE::RWKV_ATT, ss, H, hh + 2, dtype};

                jobs20 = (size_t)&job3;
                auto job4 = MatMulJob{nullptr, out, ww, kk, vv, uu, rr, T, B, C / H, bb, JOBTYPE::RWKV_ATT, ss, H, hh + 3, dtype};

                jobs22 = (size_t)&job4;
                auto job5 = MatMulJob{nullptr, out, ww, kk, vv, uu, rr, T, B, C / H, bb, JOBTYPE::RWKV_ATT, ss, H, hh + 4, dtype};

                jobs30 = (size_t)&job5;
                auto job6 = MatMulJob{nullptr, out, ww, kk, vv, uu, rr, T, B, C / H, bb, JOBTYPE::RWKV_ATT, ss, H, hh + 5, dtype};

                jobs32 = (size_t)&job6;
                auto job7 = MatMulJob{nullptr, out, ww, kk, vv, uu, rr, T, B, C / H, bb, JOBTYPE::RWKV_ATT, ss, H, hh + 6, dtype};

                jobs40 = (size_t)&job7;
                auto job8 = MatMulJob{nullptr, out, ww, kk, vv, uu, rr, T, B, C / H, bb, JOBTYPE::RWKV_ATT, ss, H, hh + 7, dtype};

                jobs42 = (size_t)&job8;

                // wait for all jobs to be done
                while (
                    (jobs10 != 0) || (jobs12 != 0) ||
                    (jobs20 != 0) || (jobs22 != 0) ||
                    (jobs30 != 0) || (jobs32 != 0) ||
                    (jobs40 != 0) || (jobs42 != 0))
                {
                }
            }
        }
}

#endif // UINT8THREADING_H