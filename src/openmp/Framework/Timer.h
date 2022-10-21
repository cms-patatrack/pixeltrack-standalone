
#include <string>
#include <vector>
#include <map>
#include <chrono>

class TimerManager;

// Using "unsigned char" gives 254 timer levels (0 is reserved)
// Use a longer type (eg. unsigned short) to increase the limit
typedef unsigned char timer_id_t;

template <int N>
class StackKeyParam {
public:
  // The union is for a performance hack
  // Use the array of small types to store the stack of timer id's.
  // Use the larger types for fast comparison for storage in a map.

  // If timer_id_t is char, there can be up to 254 timers.
  // N is the number of long ints to store timer nesting levels.
  // Each N gives (64 bits/long int) / (8 bits/char) = 8 levels
  union {
    long int   long_buckets[N];
    timer_id_t short_buckets[sizeof( long int ) * N / sizeof( timer_id_t )];
  };

  static const int max_level = sizeof( long int ) * N / sizeof( timer_id_t );

  int level;

  StackKeyParam() : level( 0 ) {
    for ( int j = 0; j < N; j++ ) { long_buckets[j] = 0; }
  }

  void add_id( timer_id_t c1 ) {
    short_buckets[level] = c1;
    level++;
  }

  timer_id_t get_id( int idx ) const { return short_buckets[idx]; }
  void       put_id( timer_id_t c1 ) { short_buckets[level] = c1; }

  bool operator==( const StackKeyParam& rhs ) const {
    bool same = false;
    for ( int j = 0; j < N; j++ ) { same &= this->long_buckets[j] == rhs.long_buckets[j]; }
    return same;
  }

  bool operator<( const StackKeyParam& rhs ) const {
    for ( int j = 0; j < N; j++ ) {
      if ( !( this->long_buckets[j] == rhs.long_buckets[j] ) ) { return this->long_buckets[j] < rhs.long_buckets[j]; }
    }
    return this->long_buckets[N - 1] < rhs.long_buckets[N - 1];
  }
};

// N = 2 gives 16 nesting levels
typedef StackKeyParam<2> StackKey;

class Timer {
  using TimePoint = std::chrono::system_clock::time_point;
  std::string m_name;
  timer_id_t  m_timer_id;

  TimePoint m_start_time;
  double m_total_time;
  int    m_num_calls;

  TimerManager* m_manager;

  StackKey m_current_stack_key;

  std::map<StackKey, double> per_stack_m_total_time;
  std::map<StackKey, int>    per_stack_m_num_calls;

public:
  void start();

  void stop();

  Timer( const std::string name, TimerManager* manager )
      : m_name( name ), m_total_time( 0.0 ), m_num_calls( 0 ), m_manager( manager ) {}

  std::string name() const { return m_name; }

  double total_time() const { return m_total_time; }

  int num_calls() const { return m_num_calls; }

  std::map<StackKey, double>& get_per_stack_time() { return per_stack_m_total_time; }
  std::map<StackKey, int>&    get_per_stack_calls() { return per_stack_m_num_calls; }

  StackKey get_stack_key() { return m_current_stack_key; }

  void       set_id( timer_id_t id ) { m_timer_id = id; }
  timer_id_t get_id() const { return m_timer_id; }
};

class TimerManager {
  std::vector<Timer*> m_timer_list;

  static thread_local std::vector<Timer*> m_current_timer_stack;

  std::map<timer_id_t, std::string> m_timer_id_to_name;
  std::map<std::string, timer_id_t> m_timer_name_to_id;

  timer_id_t max_timer_id;

public:
  TimerManager() : max_timer_id( 1 ) {}

  Timer* createTimer( const std::string& name ) {
    Timer* timer = new Timer( name, this );
    if ( m_timer_name_to_id.find( name ) == m_timer_name_to_id.end() ) {
      timer->set_id( max_timer_id );
      m_timer_id_to_name[max_timer_id] = name;
      m_timer_name_to_id[name]         = max_timer_id;
      max_timer_id++;
    } else {
      timer->set_id( m_timer_name_to_id[name] );
    }
    m_timer_list.push_back( timer );
    return timer;
  }

  Timer* current_timer() {
    Timer* current = nullptr;
    if ( m_current_timer_stack.size() > 0 ) { current = m_current_timer_stack.back(); }
    return current;
  }

  void push_timer( Timer* t ) { m_current_timer_stack.push_back( t ); }

  void pop_timer() { m_current_timer_stack.pop_back(); }

  std::string get_stack_name();

  std::string             get_name_from_id( timer_id_t );
  std::vector<timer_id_t> get_stack_key_as_vector( const StackKey& key );
  std::vector<std::string> get_stack_key_as_string_vector( const StackKey& key );

  void print_flat_profile();
  void print_stack_profile();
};

extern TimerManager timerManager;

