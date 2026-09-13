#pragma once

#include <cstdint>
#include <exception>
#include <stdexcept>

#if defined(_WIN32)
#define GFFX_CUDA_DRIVER_CALL __stdcall
#else
#define GFFX_CUDA_DRIVER_CALL
#endif

// Private adapter utility, not a public ABI. All functions come from an already-loaded
// CUDA driver. The injected table permits lifecycle testing without linking CUDA/Torch.
namespace gffx_torch_cuda {
using Context=void*;
struct Driver {
    int (GFFX_CUDA_DRIVER_CALL *pointer_get_attribute)(void*,int,unsigned long long)=nullptr;
    int (GFFX_CUDA_DRIVER_CALL *get_current)(Context*)=nullptr;
    int (GFFX_CUDA_DRIVER_CALL *device_get)(int*,int)=nullptr;
    int (GFFX_CUDA_DRIVER_CALL *get_device)(int*)=nullptr;
    int (GFFX_CUDA_DRIVER_CALL *push_current)(Context)=nullptr;
    int (GFFX_CUDA_DRIVER_CALL *pop_current)(Context*)=nullptr;
    int (GFFX_CUDA_DRIVER_CALL *set_current)(Context)=nullptr;
    int (GFFX_CUDA_DRIVER_CALL *primary_get_state)(int,unsigned*,int*)=nullptr;
    int (GFFX_CUDA_DRIVER_CALL *primary_retain)(Context*,int)=nullptr;
    int (GFFX_CUDA_DRIVER_CALL *primary_release)(int)=nullptr;
};

class Scope {
public:
    Scope(const Driver& driver,int device_index,const void* pointer,bool nonempty):driver_(driver) {
        if(!driver_.get_current || !driver_.device_get || !driver_.get_device ||
           !driver_.push_current || !driver_.pop_current || !driver_.set_current ||
           (nonempty && !driver_.pointer_get_attribute) ||
           (!nonempty && (!driver_.primary_get_state || !driver_.primary_retain || !driver_.primary_release)))
            throw std::runtime_error("CUDA context scope driver functions unavailable");
        try {
            if(driver_.get_current(&previous_)!=0)
                throw std::runtime_error("CUDA context scope cannot read previous context");
            previous_known_=true;
            if(device_index<0 || driver_.device_get(&device_,device_index)!=0)
                throw std::runtime_error("CUDA context scope device is unavailable");
            if(nonempty) {
                if(pointer==nullptr || driver_.pointer_get_attribute(&owner_,1,
                    static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(pointer)))!=0 || !owner_)
                    throw std::runtime_error("CUDA tensor owning context is unavailable or ambiguous");
            } else {
                unsigned flags=0;int active=0;
                // GetState itself does not create a primary context. Never Retain an
                // inactive one. Concurrent owner reset/destruction violates caller lifetime.
                if(driver_.primary_get_state(device_,&flags,&active)!=0 || active!=1)
                    throw std::runtime_error("empty CUDA tensor requires an already-active primary context");
                if(driver_.primary_retain(&owner_,device_)!=0)
                    throw std::runtime_error("CUDA context scope could not retain active primary context");
                retained_=true;
                if(!owner_)throw std::runtime_error("active CUDA primary context is ambiguous");
            }
            push_attempted_=true;
            if(driver_.push_current(owner_)!=0)
                throw std::runtime_error("CUDA context scope could not push tensor owner");
            pushed_=true;
            Context current=nullptr;int actual_device=-1;
            if(driver_.get_current(&current)!=0 || current!=owner_ ||
               driver_.get_device(&actual_device)!=0 || actual_device!=device_)
                throw std::runtime_error("CUDA tensor owning context does not match requested device");
        } catch(...) {
            const auto original=std::current_exception();
            if(const char* failure=close())throw std::runtime_error(failure);
            std::rethrow_exception(original);
        }
    }
    Scope(const Scope&)=delete;
    Scope& operator=(const Scope&)=delete;
    ~Scope() noexcept { close(); }

    // Checked on the normal path. Cleanup failures remain failures even if fallback
    // restores the visible current context; never advertise a successful operation.
    const char* close() noexcept {
        if(closed_)return error_;
        closed_=true;
        if(pushed_) {
            Context popped=nullptr;
            if(driver_.pop_current(&popped)!=0 || popped!=owner_)
                error_="CUDA context scope pop failed or returned another context";
            pushed_=false;
        }
        if(previous_known_ && push_attempted_) {
            Context now=nullptr;
            const bool restored=driver_.get_current(&now)==0 && now==previous_;
            if(!restored || error_) {
                if(!error_)error_="CUDA context scope did not restore previous context";
                if(driver_.set_current(previous_)!=0)
                    error_="CUDA context scope restoration failed";
                else {
                    Context verify=nullptr;
                    if(driver_.get_current(&verify)!=0 || verify!=previous_)
                        error_="CUDA context scope fallback restoration could not be verified";
                }
            }
        }
        if(retained_) {
            retained_=false;
            if(driver_.primary_release(device_)!=0)
                error_="CUDA context scope primary release failed";
        }
        return error_;
    }
private:
    Driver driver_;
    Context previous_=nullptr,owner_=nullptr;
    int device_=-1;
    bool previous_known_=false,push_attempted_=false,pushed_=false,retained_=false,closed_=false;
    const char* error_=nullptr;
};
} // namespace gffx_torch_cuda
