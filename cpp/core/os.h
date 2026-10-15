#ifndef CORE_OS_H_
#define CORE_OS_H_

#if defined(_WIN32) || defined(_WIN64)
#define OS_IS_WINDOWS
#elif defined(__unix) || defined(__APPLE__)
#define OS_IS_UNIX_OR_APPLE
#elif defined(__wasi__)
//Enclosure fork: the WebAssembly build (games/enclosure/wasm). WASI's C library has the POSIX calls the Unix paths use (time,
//files); OS_IS_WASI marks what differs.
#define OS_IS_UNIX_OR_APPLE
#define OS_IS_WASI
#else
#error Unknown operating system!
#endif

#endif //CORE_OS_H_
