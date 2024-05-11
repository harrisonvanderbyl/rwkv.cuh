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
class ThreadStream
{
public:
    std::vector<std::function<void()>> jobs;
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
        this->cv.notify_all();
        this->thread->join();
    }


    void add_job(std::function<void()> job)
    {
        std::unique_lock<std::mutex> lck(this->mtx);
        this->jobs.push_back(job);
        this->cv.notify_all();
    }

    void run()
    {
        while (!this->done)
        {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lck(this->mtx);
                this->cv.wait(lck, [this]
                              { return !this->jobs.empty() || this->done; });
                if (this->jobs.empty())
                {
                    continue;
                }
                job = this->jobs.front();
                this->jobs.erase(this->jobs.begin());
            }
            job();
        }
    }
};

class ThreadPool
{
public:
    std::vector<ThreadStream> streams;

    size_t heads = 0;
    void sync(bool block = false)
    {
        
        // create a shared atomic variable
        std::atomic<int>& counter = *(new std::atomic<int>(0));
        std::condition_variable cv;

        
        auto size = this->streams.size();
        for (size_t i = 0; i < this->streams.size(); i++)
        {
            this->streams[i].add_job([ i,&counter, size]()
                                     {
                                         
                                         counter.fetch_add(1);
                                         while (counter.load() < size)
                                         {
                                             std::this_thread::yield();
                                         }
                                     });
        }

        // if (block)
        // {
        //     std::unique_lock<std::mutex> lck(mtx);
        //     cv.wait(lck, [&]
        //             { return counter->load() == this->streams.size(); });
        // }

        if (block){

            std::cout << "Hard sync, please only use in testing!" << std::endl;
            while (counter.load() != this->streams.size()){
                std::this_thread::sleep_for(std::chrono::microseconds(18));
            
            }

        }
    }
    ThreadPool(size_t threadsNum)
    {
        heads = threadsNum;
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
};

ThreadPool *get_threadpool(size_t threadsNum = 0);

#endif // TENSOR_OPERATORS_THREADING_THREADING_H_