// threading

#ifndef TENSOR_OPERATORS_THREADING_THREADING_H_
#define TENSOR_OPERATORS_THREADING_THREADING_H_

#include <atomic>
#include <thread>
#include <iostream>
#include <vector>
#include <functional>
#include <mutex>
#include <condition_variable>

// each threadstream has a separate list of jobs
class JobLinkItem{
    std::function<void()> job;
    
    public:
    std::atomic<JobLinkItem*> next = 0;
    void addJob(JobLinkItem* job){
        auto expect = (JobLinkItem*)nullptr;
        if(! next.compare_exchange_strong(expect,job)){
            next.load()->addJob(job);
        }
    }
    JobLinkItem(std::function<void()> job){
        this->job = job;
    }

    JobLinkItem* doJob(){
        this->job();

        while(!this->next.load()){
            // std::this_thread::sleep_for(std::chrono::microseconds(1));
            std::this_thread::yield();
        }
        return this->next.load();
    }
};
class ThreadStream
{
public:
    std::atomic<JobLinkItem*> joblist = nullptr;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> done = false;
    std::atomic<bool> running = false;
    std::thread *thread = nullptr;

    ThreadStream(ThreadStream &&)
    {
    }
    ThreadStream()
    {
    }

    void start()
    {
        this->running = true;
        this->thread = new std::thread(&ThreadStream::run, this);
    }

    void stop()
    {
        this->done = true;
        this->thread->join();
    }

    void add_job(std::function<void()> job)
    {
        auto createdJob = new JobLinkItem(job);

        this->joblist.load()->addJob(createdJob);

    }

    void run()
    {
        this->joblist.store(
            new JobLinkItem([](){
                std::cout << "Thread started";
            }));

        auto next = this->joblist.load()->doJob();

        while(next){

            joblist.store(next);

            next = next->doJob();

        }
    }
};

class ThreadPool
{
public:
    std::vector<ThreadStream> streams;

    size_t heads = 0;
    bool debugmode = false;
    void sync(bool block = false)
    {

        // create a shared atomic variable
        std::atomic<int> &counter = *(new std::atomic<int>(0));
        std::condition_variable cv;

        auto size = this->streams.size();
        for (size_t i = 0; i < this->streams.size(); i++)
        {
            this->streams[i].add_job([i, &counter, size, block]()
                                     {
                                        if (i == 0 && block){

            std::cout << "Hard sync, please only use in testing!" << std::endl;
                                     }
                                    
                                         
                                         counter.fetch_add(1);
                                         while (counter.load() < size)
                                         {
                                             std::this_thread::yield();
                                         } });
        }

        // if (block)
        // {
        //     std::unique_lock<std::mutex> lck(mtx);
        //     cv.wait(lck, [&]
        //             { return counter->load() == this->streams.size(); });
        // }

        if (block)
        {

            while (counter.load() != this->streams.size())
            {
                std::this_thread::sleep_for(std::chrono::microseconds(18));
            }
        }
    }
    ThreadPool(size_t threadsNum, bool debug = false)
    {
        heads = threadsNum;
        this->debugmode = debug;
        for (size_t i = 0; i < threadsNum; i++)
        {
            this->streams.push_back(ThreadStream());
        }
    }

    void add_job(std::function<void()> job, size_t stream)
    {
        this->streams[stream % this->streams.size()].add_job(job);
    }

    void start()
    {
        for (size_t i = 0; i < this->streams.size(); i++)
        {
            this->streams[i].start();
        }
    }

    void stop()
    {
        for (size_t i = 0; i < this->streams.size(); i++)
        {
            this->streams[i].stop();
        }
    }

    void debug(Tensor t, std::string message = "")
    {
        
        if (!debugmode)
        {
            return;
        }


        sync(true);

        if (t.device == DEVICE::CUDA)
        {
            std::cout << message << std::endl;
            std::cout << t << std::endl;
            
        }
        else{
        // create a print job
        auto job = [t, message]
        {
            std::cout << message << std::endl;
            std::cout << t << std::endl;
        };

        // add the job to the first stream
        this->streams[0].add_job(job);
        }
        sync(true);
    }

    void print(std::string message)
    {
        // create a print job
        auto job = [message]
        {
            std::cout << message;
            std::cout << std::flush;
        };

        // add the job to the first stream
        this->streams[0].add_job(job);

    }
};

ThreadPool *get_threadpool(size_t threadsNum = 0, bool debug = false);

#endif // TENSOR_OPERATORS_THREADING_THREADING_H_