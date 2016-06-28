#ifndef UTIL_GLOBAL_H
#define UTIL_GLOBAL_H

//-----------------------------------------------------------------------------
// Debugging and Assertions 

#define UTIL_DEBUG

#ifndef UTIL_DEBUG
#define NDEBUG        
#endif

#include "assert.h"

//-----------------------------------------------------------------------------
// Exception Macros

#include <iostream>
#define UTIL_THROW(msg) {std::cout << msg; throw 1;}

#if 0
#include "Exception.h"
/**
* Macro for the name of the current function (compiler dependent).
*/
#define UTIL_FUNC __PRETTY_FUNCTION__

/**
* Macro for throwing an Exception, reporting function, file and line number.
*/
#ifdef  UTIL_FUNC
  #define UTIL_THROW(msg) throw Exception(UTIL_FUNC, msg, __FILE__, __LINE__)
#else
  #define UTIL_THROW(msg) throw Exception(msg, __FILE__, __LINE__)
#endif

/**
* Assertion macro suitable for serial or parallel production code.
*/
#define UTIL_CHECK(condition) \
  if (!(condition)) { UTIL_THROW("Failed assertion: " #condition); }

/**
* Assertion macro suitable for debugging serial or parallel code.
*/
#ifdef NDEBUG
#define UTIL_ASSERT(condition) {}
#else
#define UTIL_ASSERT(condition) \
  if (!(condition)) { UTIL_THROW("Failed assertion: " #condition); }
#endif

#endif  // if 0

#endif
