/* Reference-only CPython loop: original Decimal arithmetic, no binary conversion. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
typedef struct {int i,j,sl,sh;PyObject *lo,*hi;} term;
static int sign(PyObject *v,PyObject *zero) {
 int e=PyObject_RichCompareBool(v,zero,Py_EQ);if(e<0)return -2;if(e)return 0;
 e=PyObject_RichCompareBool(v,zero,Py_GT);if(e<0)return -2;return e?1:-1;
}
static int terms(PyObject *rows,term *out,int order,PyObject *zero) {
 if(!PyList_Check(rows)||PyList_GET_SIZE(rows)>22){PyErr_SetString(PyExc_ValueError,"bounded term list required");return -1;}
 int n=(int)PyList_GET_SIZE(rows);uint32_t seen=0;
 for(int k=0;k<n;k++) {
  PyObject *v=PyList_GET_ITEM(rows,k);if(!PyTuple_Check(v)||PyTuple_GET_SIZE(v)!=4){PyErr_SetString(PyExc_ValueError,"term tuple required");return -1;}
  long i=PyLong_AsLong(PyTuple_GET_ITEM(v,0)),j=PyLong_AsLong(PyTuple_GET_ITEM(v,1));if(PyErr_Occurred())return -1;
  if(i<0||i>order||j<0||j>1||(seen&(1u<<(2*i+j)))){PyErr_SetString(PyExc_ValueError,"unique bounded coefficient required");return -1;}seen|=1u<<(2*i+j);
  out[k].i=(int)i;out[k].j=(int)j;out[k].lo=PyTuple_GET_ITEM(v,2);out[k].hi=PyTuple_GET_ITEM(v,3);
  out[k].sl=sign(out[k].lo,zero);out[k].sh=sign(out[k].hi,zero);if(out[k].sl==-2||out[k].sh==-2)return -1;
 }
 return n;
}
static PyObject *extreme(term *a,term *b,int *pairs,int n,int upper) {
 PyObject *av[2]={a->lo,a->hi},*bv[2]={b->lo,b->hi};PyObject *best=NULL;
 for(int k=0;k<n;k++) {
  PyObject *v=PyNumber_Multiply(av[pairs[k]/2],bv[pairs[k]%2]);if(!v){Py_XDECREF(best);return NULL;}
  if(!best){best=v;continue;}
  int take=PyObject_RichCompareBool(v,best,upper?Py_GT:Py_LT);if(take<0){Py_DECREF(best);Py_DECREF(v);return NULL;}
  if(take){Py_DECREF(best);best=v;}else Py_DECREF(v);
 }
 return best;
}
static PyObject *convolve(PyObject *self,PyObject *args) {
 PyObject *aa,*bb,*ctx,*zero,*floor,*ceil,*result=NULL;int order;
 if(!PyArg_ParseTuple(args,"OOiOOOO",&aa,&bb,&order,&ctx,&zero,&floor,&ceil))return NULL;
 if(order<0||order>10){PyErr_SetString(PyExc_ValueError,"order0..10 required");return NULL;}
 term a[22],b[22];int na=terms(aa,a,order,zero),nb=terms(bb,b,order,zero);if(na<0||nb<0)return NULL;
 PyObject *lo[22]={0},*hi[22]={0};int keys[22],count=0;
 for(int x=0;x<na;x++)for(int y=0;y<nb;y++) {
  int i=a[x].i+b[y].i,j=a[x].j+b[y].j;if(i>order||j>1)continue;int key=2*i+j;
  PyObject *l=NULL,*h=NULL;
  if((a[x].sl==0&&a[x].sh==0)||(b[y].sl==0&&b[y].sh==0)){l=Py_NewRef(zero);h=Py_NewRef(zero);}
  else {
   int lp[4]={0},hp[4]={0},nl=1,nh=1;term *u=&a[x],*v=&b[y];
   if(!u->sl||!u->sh||!v->sl||!v->sh){nl=nh=4;for(int k=0;k<4;k++)lp[k]=hp[k]=k;}
   else if(u->sl>0){if(v->sl>0){lp[0]=0;hp[0]=3;}else if(v->sh<0){lp[0]=2;hp[0]=1;}else{lp[0]=2;hp[0]=3;}}
   else if(u->sh<0){if(v->sl>0){lp[0]=1;hp[0]=2;}else if(v->sh<0){lp[0]=3;hp[0]=0;}else{lp[0]=1;hp[0]=0;}}
   else if(v->sl>0){lp[0]=1;hp[0]=3;}
   else if(v->sh<0){lp[0]=2;hp[0]=0;}
   else{nl=nh=2;lp[0]=1;lp[1]=2;hp[0]=0;hp[1]=3;}
   if(PyObject_SetAttrString(ctx,"rounding",floor)<0)goto fail;l=extreme(u,v,lp,nl,0);if(!l)goto fail;
   if(PyObject_SetAttrString(ctx,"rounding",ceil)<0){Py_DECREF(l);goto fail;}h=extreme(u,v,hp,nh,1);if(!h){Py_DECREF(l);goto fail;}
  }
  if(lo[key]) {
   int zl=PyObject_RichCompareBool(lo[key],zero,Py_EQ),zh=PyObject_RichCompareBool(hi[key],zero,Py_EQ);
   if(zl<0||zh<0){Py_DECREF(l);Py_DECREF(h);goto fail;}
   if(!(zl&&zh)) {
    int ml=PyObject_RichCompareBool(l,zero,Py_EQ),mh=PyObject_RichCompareBool(h,zero,Py_EQ);
    if(ml<0||mh<0){Py_DECREF(l);Py_DECREF(h);goto fail;}
    if(ml&&mh){Py_DECREF(l);Py_DECREF(h);continue;}
    if(PyObject_SetAttrString(ctx,"rounding",floor)<0){Py_DECREF(l);Py_DECREF(h);goto fail;}
    PyObject *v=PyNumber_Add(lo[key],l);Py_DECREF(l);l=v;if(!l){Py_DECREF(h);goto fail;}
    if(PyObject_SetAttrString(ctx,"rounding",ceil)<0){Py_DECREF(l);Py_DECREF(h);goto fail;}
    v=PyNumber_Add(hi[key],h);Py_DECREF(h);h=v;if(!h){Py_DECREF(l);goto fail;}
   }
   Py_DECREF(lo[key]);Py_DECREF(hi[key]);
  }else keys[count++]=key;
  lo[key]=l;hi[key]=h;
 }
 result=PyDict_New();if(!result)goto fail;
 for(int k=0;k<count;k++) {
  int key=keys[k];PyObject *index=Py_BuildValue("(ii)",key/2,key%2),*value=PyTuple_Pack(2,lo[key],hi[key]);
  if(!index||!value||PyDict_SetItem(result,index,value)<0){Py_XDECREF(index);Py_XDECREF(value);Py_CLEAR(result);goto fail;}Py_DECREF(index);Py_DECREF(value);
 }
fail:
 for(int k=0;k<22;k++){Py_XDECREF(lo[k]);Py_XDECREF(hi[k]);}
 return result;
}
static PyMethodDef methods[]={{"convolve",convolve,METH_VARARGS,"Original directed Decimal convolution."},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_opaque_decimal_native",NULL,-1,methods};
PyMODINIT_FUNC PyInit__opaque_decimal_native(void){return PyModule_Create(&module);}
