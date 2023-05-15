#if defined(_WIN32) && !defined(__MINGW32__)
#define MY_API __declspec(dllexport)
#else
#define MY_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

MY_API int sayHello();
MY_API int freeMem();

#ifdef __cplusplus
}
#endif
