#if defined(_WIN32) && !defined(__MINGW32__)
#define MY_API __declspec(dllexport)
#else
#define MY_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

MY_API void* getContainer();
MY_API int sayHello(void* container);
MY_API int freeMem(void* container);

#ifdef __cplusplus
}
#endif
