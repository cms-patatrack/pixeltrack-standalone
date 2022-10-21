
#include "Framework/Profile.h"
#include "Framework/Event.h"
#include "Framework/Timer.h"
#include <tbb/tbb.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <mutex>
#include <map>

#include <ittnotify.h>


using TimePoint = std::chrono::system_clock::time_point;

TimePoint program_start = std::chrono::system_clock::now();

const int MAX_EVENTS = 1000;
TimePoint produce_start[MAX_EVENTS];
TimePoint acquire_start[MAX_EVENTS];

struct EventRecord
{
    double timestamp;
    int event_id;
    char event_type;
    std::string name;
    std::thread::id tid;
    std::string category;
    double duration;

    EventRecord(double ts, int id, char type, const std::string& ename, std::thread::id id1, const std::string& cat, double dur=0.0) : timestamp(ts), event_id(id), event_type(type), name(ename), tid(id1), category(cat), duration(dur) {}
};

std::vector<EventRecord> events;
std::mutex event_mutex;

thread_local std::map<std::string, Timer *> timerMap;
//std::map<std::string, Timer *> timerMap;
//std::map<std::pair<std::string, int>, Timer *> timerMap;
//std::map<std::string, Timer *> timerMap;

void dumpEventRecord()
{
    std::ofstream f("events.json");
    f << "[\n";
    for (auto &er : events) {
        f << R"({"name":")" << er.name << R"(",)" <<
             R"("ph":")" << er.event_type << R"(",)" <<
             R"("cat":")" << er.category << R"(",)" <<
             R"("pid":")" << 1 << R"(",)" <<
             R"("tid":")" << er.tid << R"(",)" <<
             R"("ts":)" << (uint64_t)(er.timestamp * 1.0e6);
        if (er.event_type == 'X')
            f << R"(,"dur":)" << (uint64_t)(er.duration * 1.0e6);
        f <<  "},\n";
        
    }
    f << "]\n";
}

__itt_domain* task_domain = __itt_domain_create("pixeltrack");
__itt_string_handle* task_name = __itt_string_handle_create("this");


void beginProduce(const std::string& name, const edm::Event& event)
{ 
    //event_mutex.lock();
    auto timer_it = timerMap.find(name);
    //auto timer_it = timerMap.find(name + std::to_string(event.eventID()));
    //auto timer_it = timerMap.find(std::make_pair(name, event.eventID()));
    //std::cout << " starting for " << name + std::to_string(event.eventID()) << std::endl;
    if (timer_it == timerMap.end()) {
        //Timer* tmp  = timerManager.createTimer(name + std::to_string(event.eventID()));
        event_mutex.lock();
        //Timer* tmp  = timerManager.createTimer(name + std::to_string(event.eventID()));
        Timer* tmp  = timerManager.createTimer(name);
        //timerMap[make_pair(name, event.eventID())]  = tmp;
        timerMap[name]  = tmp;
        event_mutex.unlock();
        tmp->start();
       
    } else {
        timer_it->second->start();
    }
    //event_mutex.unlock();

    __itt_id task_id = __itt_null;
    __itt_id parent_task = __itt_null;
    __itt_string_handle* task_name = __itt_string_handle_create(name.c_str());
    __itt_task_begin(task_domain, task_id, parent_task, task_name);
#if 0
    produce_start[event.eventID()] = std::chrono::system_clock::now();
    std::chrono::duration<double> from_program_start = produce_start[event.eventID()] - program_start;
    //std::chrono::duration<double> from_program_start = produce_start[event.eventID()] - acquire_start[1];
    std::thread::id tid = std::this_thread::get_id();
    std::cout << "beginProduce for " << name << " event = " << event.eventID() <<  " time = " << from_program_start.count() << " on thread " << tid << std::endl;

    //events.push_back(EventRecord(from_program_start.count(), event.eventID(), 'B', name, tid));
#endif
}
void endProduce(const std::string& name, const edm::Event& event)
{
    //auto timer_it = timerMap.find(name + std::to_string(event.eventID()));
    auto timer_it = timerMap.find(name);
    //event_mutex.lock();
    //auto timer_it = timerMap.find(std::make_pair(name, event.eventID()));
    //event_mutex.unlock();
    if (timer_it == timerMap.end()) {
        //assert(false);
        std::cout << "Bad!! Timer not found for " << name + std::to_string(event.eventID()) << std::endl;
    } else {
        timer_it->second->stop();
    }
    __itt_task_end(task_domain);
#if 0
    TimePoint produce_end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = produce_end - produce_start[event.eventID()];
    std::chrono::duration<double> from_program_start = produce_end - program_start;
    //std::chrono::duration<double> from_program_start = produce_end - acquire_start[1];
    std::chrono::duration<double> begin_from_program_start = produce_start[event.eventID()] - program_start;
    //std::chrono::duration<double> begin_from_program_start = produce_start[event.eventID()] - acquire_start[1];
    std::thread::id tid = std::this_thread::get_id();
    std::cout << "endProduce for " << name << " event = " << event.eventID() << " start time = " << begin_from_program_start.count() << " produce duration = " << elapsed.count() << std::endl;
    event_mutex.lock();
    //events.push_back(EventRecord(from_program_start.count(), event.eventID(), 'E', name, tid));
   
    events.push_back(EventRecord(begin_from_program_start.count(), event.eventID(), 'X', name, tid, "produce", elapsed.count()));
    event_mutex.unlock();
#endif
}

void beginAcquire(const std::string& name, const edm::Event& event)
{
#if 0
    acquire_start[event.eventID()] = std::chrono::system_clock::now();
    std::chrono::duration<double> from_program_start = acquire_start[event.eventID()] - program_start;
    //std::chrono::duration<double> from_program_start = acquire_start[event.eventID()] - acquire_start[1];
    std::cout << "beginAcquire for " << name << " event = " << event.eventID() << " time = " << from_program_start.count() << std::endl;
#endif
}
void endAcquire(const std::string& name, const edm::Event& event) 
{
#if 0
    TimePoint acquire_end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = acquire_end - acquire_start[event.eventID()];
    std::chrono::duration<double> from_program_start = acquire_end - program_start;
    //std::chrono::duration<double> from_program_start = acquire_end - acquire_start[1];
    std::chrono::duration<double> begin_from_program_start = acquire_start[event.eventID()] - program_start;
    //std::chrono::duration<double> begin_from_program_start = acquire_start[event.eventID()] - acquire_start[1];
    std::cout << "endAcquire for " << name << " event = " << event.eventID() << " start time = " << begin_from_program_start.count() << " acquire duration = " << elapsed.count() << std::endl;
    std::thread::id tid = std::this_thread::get_id();
    event_mutex.lock();

    events.push_back(EventRecord(begin_from_program_start.count(), event.eventID(), 'X', name, tid, "acquire", elapsed.count()));
    event_mutex.unlock();
#endif
}
  
