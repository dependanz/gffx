#include "../../adapters/pytorch/cuda_context_scope.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using gffx_torch_cuda::Driver;
using gffx_torch_cuda::Scope;
using Context=gffx_torch_cuda::Context;
#define CHECK(x) do { if (!(x)) { std::cerr << "failed line " << __LINE__ << ": " #x << '\n'; std::exit(1); } } while (0)

namespace {
struct Fake {
    Context current=reinterpret_cast<Context>(2),owner=reinterpret_cast<Context>(1);
    std::vector<Context> stack;
    std::string fail;
    bool active=true,null_owner=false,wrong_device=false;
    int retains=0,releases=0,pushes=0,pops=0,sets=0,states=0,pointers=0,gets=0;
} f;

int GFFX_CUDA_DRIVER_CALL pointer(void *out,int attribute,unsigned long long address) {
    ++f.pointers;CHECK(attribute==1 && address==42);
    if(f.fail=="pointer")return 1;
    *static_cast<Context*>(out)=f.null_owner?nullptr:f.owner;return 0;
}
int GFFX_CUDA_DRIVER_CALL current(Context *out) {
    ++f.gets;
    if(f.fail=="get" || (f.fail=="get_after_push" && f.pushes && f.gets==2))return 1;
    *out=f.current;return 0;
}
int GFFX_CUDA_DRIVER_CALL device(int *out,int ordinal) {
    if(f.fail=="device")return 1;*out=ordinal;return 0;
}
int GFFX_CUDA_DRIVER_CALL owner_device(int *out) {
    if(f.fail=="owner_device")return 1;*out=f.wrong_device?1:0;return 0;
}
int GFFX_CUDA_DRIVER_CALL push(Context value) {
    ++f.pushes;if(f.fail=="push")return 1;
    f.stack.push_back(f.current);f.current=value;return 0;
}
int GFFX_CUDA_DRIVER_CALL pop(Context *out) {
    ++f.pops;if(f.fail=="pop" || f.fail=="pop_and_set")return 1;
    CHECK(!f.stack.empty());*out=f.current;f.current=f.stack.back();f.stack.pop_back();return 0;
}
int GFFX_CUDA_DRIVER_CALL set(Context value) {
    ++f.sets;if(f.fail=="pop_and_set")return 1;
    f.current=value;return 0;
}
int GFFX_CUDA_DRIVER_CALL state(int,unsigned *flags,int *active) {
    ++f.states;if(f.fail=="state")return 1;*flags=0;*active=f.active?1:0;return 0;
}
int GFFX_CUDA_DRIVER_CALL retain(Context *out,int) {
    if(f.fail=="retain")return 1;++f.retains;*out=f.null_owner?nullptr:f.owner;return 0;
}
int GFFX_CUDA_DRIVER_CALL release(int) {
    ++f.releases;return f.fail=="release"?1:0;
}
Driver driver() { return Driver{pointer,current,device,owner_device,push,pop,set,state,retain,release}; }
bool rejected(const Driver& d,bool nonempty=true) {
    try { Scope scope(d,0,nonempty?reinterpret_cast<void*>(42):nullptr,nonempty); }
    catch(const std::runtime_error&) { return true; }
    return false;
}
}

int main() {
    f=Fake{};
    { Scope scope(driver(),0,reinterpret_cast<void*>(42),true);
      CHECK(f.current==f.owner && f.retains==0 && f.states==0);
      CHECK(scope.close()==nullptr);CHECK(scope.close()==nullptr); }
    CHECK(f.current==reinterpret_cast<Context>(2) && f.pushes==1 && f.pops==1 && f.releases==0);

    f=Fake{};f.current=nullptr;
    { Scope scope(driver(),0,reinterpret_cast<void*>(42),true);CHECK(f.current==f.owner); }
    CHECK(f.current==nullptr && f.retains==0 && f.releases==0);

    f=Fake{};
    { Scope scope(driver(),0,nullptr,false);CHECK(f.states==1 && f.retains==1 && f.pointers==0); }
    CHECK(f.releases==1 && f.current==reinterpret_cast<Context>(2));
    f=Fake{};f.active=false;CHECK(rejected(driver(),false));CHECK(f.retains==0 && f.pushes==0);
    f=Fake{};f.null_owner=true;CHECK(rejected(driver()));CHECK(f.pushes==0 && f.retains==0);
    f=Fake{};f.null_owner=true;CHECK(rejected(driver(),false));CHECK(f.retains==1 && f.releases==1);

    for(const char *failure:{"get","device","pointer","push","get_after_push","owner_device"}) {
        f=Fake{};f.fail=failure;CHECK(rejected(driver()));
        CHECK(f.current==reinterpret_cast<Context>(2) && f.retains==f.releases);
    }
    for(const char *failure:{"state","retain","push","get_after_push","owner_device"}) {
        f=Fake{};f.fail=failure;CHECK(rejected(driver(),false));
        CHECK(f.current==reinterpret_cast<Context>(2) && f.retains==f.releases);
    }
    f=Fake{};f.wrong_device=true;CHECK(rejected(driver()));CHECK(f.pops==1 && f.current==reinterpret_cast<Context>(2));
    f=Fake{};Driver missing=driver();missing.push_current=nullptr;
    CHECK(rejected(missing));CHECK(f.gets==0 && f.pushes==0);

    f=Fake{};
    { Scope scope(driver(),0,nullptr,false);f.fail="pop";CHECK(scope.close()!=nullptr);
      CHECK(scope.close()!=nullptr && f.pops==1 && f.sets==1 && f.releases==1); }
    CHECK(f.current==reinterpret_cast<Context>(2));
    f=Fake{};
    { Scope scope(driver(),0,nullptr,false);f.fail="release";CHECK(scope.close()!=nullptr); }
    CHECK(f.pops==1 && f.releases==1 && f.current==reinterpret_cast<Context>(2));
    f=Fake{};
    { Scope scope(driver(),0,reinterpret_cast<void*>(42),true);f.fail="pop_and_set";CHECK(scope.close()!=nullptr); }
    CHECK(f.pops==1 && f.sets==1); // Restoration failure is reported, never advertised as success.
    f=Fake{};
    try { Scope scope(driver(),0,nullptr,false);throw std::logic_error("consumer error"); }
    catch(const std::logic_error&) {}
    CHECK(f.current==reinterpret_cast<Context>(2) && f.retains==1 && f.releases==1 && f.pops==1);
    std::cout << "CUDA context scope fake lifecycle checks passed\n";
    return 0;
}
