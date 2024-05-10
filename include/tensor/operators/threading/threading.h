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
        void start();
        void stop();
        void add_job(std::function<void()> job);
        void run();
        ThreadStream(ThreadStream&&){
        }
        ThreadStream() {
        }
};

class ThreadPool
{
    public:
        std::vector<ThreadStream> streams;
        ThreadPool(size_t threadsNum);
        void add_job(std::function<void()> job, size_t stream = 0);
        void start();
        void stop();
        std::function<void()> sync(){
            // shared synclock, last thread to reach, unlocks all threads
            std::atomic<size_t>& count = *new std::atomic<size_t>(0);
            std::condition_variable& cv = *new std::condition_variable();
            std::mutex& mtx = *new std::mutex();
            

            for (size_t i = 0; i < this->streams.size(); i++)
            {
                this->streams[i].add_job([&](){
                    count.fetch_add(1);

                    if (count.load() == this->streams.size()){
                        cv.notify_all();
                    }
                    else{
                        std::unique_lock<std::mutex> lck(mtx);
                        cv.wait(lck);
                    }
                });
            }

            // cv.notify_all();

            
            return [&](){
                std::unique_lock<std::mutex> lck(mtx);
                cv.wait(lck, [&] { return count.load() == this->streams.size(); });
            };
            
        }
};


void ThreadStream::start()
{
    this->running = true;
    this->thread = new std::thread(&ThreadStream::run, this);
}

void ThreadStream::stop()
{
    this->done = true;
    this->cv.notify_all();
    this->thread->join();
}

void ThreadStream::add_job(std::function<void()> job)
{
    std::unique_lock<std::mutex> lck(this->mtx);
    this->jobs.push_back(job);
    this->cv.notify_all();
}

void ThreadStream::run()
{
    while (!this->done)
    {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lck(this->mtx);
            this->cv.wait(lck, [this] { return !this->jobs.empty() || this->done; });
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

ThreadPool::ThreadPool(size_t threadsNum)
{
    for (size_t i = 0; i < threadsNum; i++)
    {
        this->streams.push_back(ThreadStream());
    }
}

void ThreadPool::add_job(std::function<void()> job, size_t stream)
{
    this->streams[stream%this->streams.size()].add_job(job);
}

void ThreadPool::start()
{
    for (size_t i = 0; i < this->streams.size(); i++)
    {
        this->streams[i].start();
    }
}

void ThreadPool::stop()
{
    for (size_t i = 0; i < this->streams.size(); i++)
    {
        this->streams[i].stop();
    }
}



#endif // TENSOR_OPERATORS_THREADING_THREADING_H_