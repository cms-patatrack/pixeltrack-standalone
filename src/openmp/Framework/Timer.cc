
#include "Timer.h"
#include <iomanip>
#include <iostream>
#include <map>

TimerManager timerManager;
thread_local std::vector<Timer*> TimerManager::m_current_timer_stack;

void Timer::start() {
  m_start_time = std::chrono::system_clock::now();
  if ( this == m_manager->current_timer() ) { std::cerr << "Timer loop: " << m_name << std::endl; }
  Timer* parent = m_manager->current_timer();
  if ( parent ) {
    m_current_stack_key = parent->get_stack_key();
    m_current_stack_key.add_id( m_timer_id );
  } else {
    m_current_stack_key = StackKey();
    m_current_stack_key.add_id( m_timer_id );
  }
  m_manager->push_timer( this );
}

void Timer::stop() {
  TimePoint now = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = now - m_start_time;
  m_total_time += elapsed.count();
  m_num_calls++;

  per_stack_m_total_time[m_current_stack_key] += elapsed.count();
  per_stack_m_num_calls[m_current_stack_key] += 1;

  if ( this != m_manager->current_timer() ) {
    std::cerr << "Timer mismatch. Start: " << m_manager->current_timer()->name() << " Stop: " << m_name << std::endl;
  }
  m_manager->pop_timer();
}

std::string TimerManager::get_stack_name() {
  std::string stack_name;
  for ( auto ti : m_current_timer_stack ) {
    stack_name += ti->name();
    stack_name += "/";
  }
  return stack_name;
}

void TimerManager::print_flat_profile() {
  size_t max_name_len = std::string( "Timer Name" ).length();
  for ( auto ti : m_timer_list ) { max_name_len = std::max( max_name_len, ti->name().length() ); }

  std::cout << std::setw( max_name_len ) << std::left << "Timer Name"
            << "       Time(s)            Calls     Time per call" << std::endl;

  for ( auto ti : m_timer_list ) {
    double time          = ti->total_time();
    int    ncalls        = ti->num_calls();
    double time_per_call = time / ncalls;

    std::cout << std::setw( max_name_len ) << std::left << ti->name() << "   " << std::fixed << std::setprecision( 6 )
              << std::setw( 15 ) << std::right << time << " " << std::setw( 12 ) << ncalls << "   "
              << std::setprecision( 6 ) << std::fixed << std::setw( 15 ) << std::right << time_per_call << std::endl;
  }
}

std::vector<std::string> split_string( const std::string& input, const char split_char ) {
  std::vector<std::string> output;

  size_t pos = 0;
  while ( pos < input.size() ) {
    size_t out_pos = input.find( split_char, pos );
    if ( out_pos == std::string::npos ) {
      output.push_back( input.substr( pos, input.size() - pos + 1 ) );
      break;
    }
    output.push_back( input.substr( pos, out_pos - pos ) );
    pos = out_pos + 1;
  }

  return output;
}

std::string join_string( const std::vector<std::string>& input ) {
  std::string out;
  for ( unsigned int i = 0; i < input.size(); i++ ) {
    out += input[i];
    if ( i != input.size() - 1 ) { out += "/"; }
  }
  return out;
}

struct ProfileData {
  double time;
  int    calls;

  ProfileData& operator+=( const ProfileData& pd ) {
    time += pd.time;
    calls += pd.calls;
    return *this;
  }
};

// Store output

struct StackProfileData {
  std::string name;
  double      incl_time;
  double      excl_time;
  int         calls;

  // Map by timer id
  //std::map<timer_id_t, StackProfileData> children;
  std::map<std::string, StackProfileData> children_string;

#if 0
  void insert( std::vector<timer_id_t>& stack_key, double time, int calls ) {
    if ( stack_key.size() == 0 ) return;

    auto spd_it = children.find( stack_key[0] );

    if ( spd_it != children.end() ) {
      if ( stack_key.size() == 1 ) {
        spd_it->second.incl_time += time;
        spd_it->second.calls  += calls;
        excl_time -= time;
      }
      std::vector<timer_id_t> rest_key( stack_key.begin() + 1, stack_key.end() );
      spd_it->second.insert( rest_key, time, calls );
    }

    if ( spd_it == children.end() ) {
      StackProfileData spd;
      spd.incl_time = time;
      spd.excl_time = time;

      excl_time -= time;
      spd.calls = calls;
      spd.name  = timerManager.get_name_from_id( stack_key[0] );
      std::vector<timer_id_t> rest_key( stack_key.begin() + 1, stack_key.end() );
      spd.insert( rest_key, time, calls );
      children[stack_key[0]] = spd;
    }
  }
#endif

  void insert( std::vector<std::string>& stack_key, double time, int calls ) {
    if ( stack_key.size() == 0 ) return;

    //std::cout << "Inserting " << stack_key[0] << std::endl;
    auto spd_it = children_string.find( stack_key[0] );

    // Next timer down the stack - already has an entry
    if ( spd_it != children_string.end() ) {
      //std::cout << "  Found, time = " << time << " excl_time = " << excl_time << std::endl;
      if ( stack_key.size() == 1 ) {
        spd_it->second.incl_time += time;
        spd_it->second.excl_time += time;
        spd_it->second.calls  += calls;
        excl_time -= time;
        //std::cout << "    new excl time = " << excl_time << std::endl;
      }
      // Insert next timer down the timer stack
      std::vector<std::string> rest_key( stack_key.begin() + 1, stack_key.end() );
      spd_it->second.insert( rest_key, time, calls );
    }

    // Next timer down the stack - no existing entry
    if ( spd_it == children_string.end() ) {
      StackProfileData spd;
      spd.incl_time = time;
      spd.excl_time = time;
      //std::cout << "  Not found, time = " << time << " fresh excl_time = " << spd.excl_time << std::endl;
      //std::cout << "     parent excl_time = " << excl_time << std::endl;

      excl_time -= time;
      //std::cout << "    new excl time = " << excl_time << std::endl;
      spd.calls = calls;
      spd.name  = stack_key[0];

      std::vector<std::string> rest_key( stack_key.begin() + 1, stack_key.end() );
      spd.insert( rest_key, time, calls );

      children_string[stack_key[0]] = spd;
    }
  }
};

int get_max_name_len( StackProfileData& spd, int indent ) {
  int indent_len   = 2;
  int max_name_len = spd.name.length() + indent_len * indent;
  for ( auto it : spd.children_string ) {
    int child_len = get_max_name_len( it.second, indent + 1 );
    max_name_len  = std::max( max_name_len, child_len );
  }
  return max_name_len;
}

void print_inner_stack( StackProfileData& spd, int indent, int max_len ) {
  std::string pad( 2 * indent, ' ' );
  std::cout << std::setw( max_len ) << std::left << pad + spd.name << "  ";
  std::cout << std::right << std::fixed << std::setprecision( 6 ) << std::setw( 15 ) << spd.incl_time << " ";
  std::cout << std::setw( 15 ) << spd.excl_time << " ";
  std::cout << std::setw( 8 ) << spd.calls;
  std::cout << std::setw( 15 ) << spd.incl_time / spd.calls << " ";
  std::cout << std::endl;
  for ( auto it : spd.children_string ) { print_inner_stack( it.second, indent + 1, max_len ); }
}

void TimerManager::print_stack_profile() {

  StackProfileData root;
  root.calls     = 0;
  root.incl_time = 0.0;
  root.excl_time = 0.0;

  for ( auto ti : m_timer_list ) {
    auto stack_time = ti->get_per_stack_time();
    for ( auto stack_it = stack_time.begin(); stack_it != stack_time.end(); stack_it++ ) {
      auto stack_key_as_vector = get_stack_key_as_string_vector( stack_it->first );
      root.insert( stack_key_as_vector, stack_it->second, ti->get_per_stack_calls()[stack_it->first] );
    }
  }

  int max_len = get_max_name_len( root, 0 );
  std::cout << std::setw( max_len ) << std::left << "Timer name"
            << "  Inclusive time "
            << "  Exclusive Time"
            << "    Calls"
            << "  Time per call" << std::endl;

  for ( auto it : root.children_string ) { print_inner_stack( it.second, 0, max_len ); }
}

std::string TimerManager::get_name_from_id( timer_id_t id ) { return m_timer_id_to_name[id]; }

std::vector<timer_id_t> TimerManager::get_stack_key_as_vector( const StackKey& key ) {
  std::vector<timer_id_t> stack_key;
  for ( int i = 0; i < StackKey::max_level; i++ ) {
    if ( key.get_id( i ) == 0 ) { break; }
    stack_key.push_back( key.get_id( i ) );
  }
  return stack_key;
}

std::vector<std::string> TimerManager::get_stack_key_as_string_vector( const StackKey& key ) {
  std::vector<std::string> stack_key;
  for ( int i = 0; i < StackKey::max_level; i++ ) {
    if ( key.get_id( i ) == 0 ) { break; }
    stack_key.push_back( get_name_from_id(key.get_id( i ) ) );
  }
  return stack_key;
}

