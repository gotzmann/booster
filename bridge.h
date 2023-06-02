#ifndef BRIDGE_H
#define BRIDGE_H

extern "C" { // ------------------------------------------------------

char * status(char * jobID);
int64_t timing(char * jobID);

}  // ------------------------------------------------------

#endif // BRIDGE_H