#ifndef BRIDGE_H
#define BRIDGE_H

#include <stdint.h>

extern "C" { // ------------------------------------------------------

const char * status(char * jobID);
int64_t timing(char * jobID);

}  // ------------------------------------------------------

#endif // BRIDGE_H